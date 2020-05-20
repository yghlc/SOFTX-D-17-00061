#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016, Erika Tudisco, Edward Andò, Stephen Hall, Rémi Cailletaud

# This file is part of TomoWarp2.
# 
# TomoWarp2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TomoWarp2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with TomoWarp2.  If not, see <http://www.gnu.org/licenses/>.

# ===================================================================
# ===========================  TomoWarp2  ===========================
# ===================================================================

# Authors: Erika Tudisco, Edward Andò, Stephen Hall, Rémi Cailletaud

# Modified from DIC_setup.py by Lingcao Huang, 20 May 2020
# In the original DIC_setup, when the node spacing is very small (such as 1), it will have a lot of nodes, and end in run out of memory errors.
# I supposed that it extract subVolume (copy voxels) for all nodes as parts of message, this step cost a lot of memory and time.
# In the DIC_setup, there is a option: memory limit, but it seems not working.
# So I try to used a better solution for parallel computing,
# Currently version may not work on multiple workstations.


""" 
INPUTS:
  - kinematics
  - "data" structure

  
OUTPUTS:
  - filled-in kinematics
"""

import sys, time
import numpy
import multiprocessing
import logging

from DIC_worker import DIC_worker
from tools.print_variable import pv

from tools.load_slices import load_slices

# This is our pixel_search C-code
from pixel_search.c_code import pixel_search
from sub_pixel.cc_interpolation import cc_interpolation_local, \
    cc_interpolation_local_2D
from sub_pixel.image_interpolation_translation_rotation import \
    image_interpolation_translation_rotation

# 2015-01-28 EA and ET: we need to pickle the DIC_worker pipes, that we are sending into a pipe to data_delivery_worker
#   See: http://stackoverflow.com/questions/1446004/python-2-6-send-connection-object-over-queue-pipe-etc
#        http://jodal.no/post/3669476502/pickling-multiprocessing-connection-objects/

# 2016-01-05 Adding full time output with code from Brian Visel's answer from:
#   http://stackoverflow.com/questions/4048651/python-function-to-convert-seconds-into-minutes-hours-and-days
intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format( int( value ), name))
    return ', '.join(result[:granularity])

import os
import psutil
process = psutil.Process(os.getpid())

import multiprocessing
from multiprocessing import Pool
import tqdm


def read_sub_volume(extent, im1, im2, zExtents_im1_current, zExtents_im2_current, data):

    nodeExtent = extent

    # Node extents are in absolute coordinates:
    #   First: Add crop in horizontal directions, -- the loaded ROI is still in absolute image coordinates
    nodeExtent[0, :, 1:3] = nodeExtent[0, :, 1:3] - data.ROI_corners[0, 0, 1:3]
    nodeExtent[1, :, 1:3] = nodeExtent[1, :, 1:3] - data.ROI_corners[1, 0, 1:3]

    # Second, add z_extent for this row of nodes.
    nodeExtent[0, :, 0] = nodeExtent[0, :, 0] - zExtents_im1_current[0]
    nodeExtent[1, :, 0] = nodeExtent[1, :, 0] - zExtents_im2_current[0]

    # # for test
    # print("DataRequest: %d, %s"%(workerNumber,(str(nodeExtent))))
    dataMessage = [None]*3
    try:

        for i_d in range(3):
            # crop the extent of requested volume to fit in the available volume (i_d = index for dimensions)
            nodeExtent[0, nodeExtent[0, :, i_d] < 0, i_d] = 0
            nodeExtent[0, nodeExtent[0, :, i_d] > im1.shape[i_d], i_d] = im1.shape[i_d]
            nodeExtent[1, nodeExtent[1, :, i_d] < 0, i_d] = 0
            nodeExtent[1, nodeExtent[1, :, i_d] > im2.shape[i_d], i_d] = im2.shape[i_d]

        im1_subvolume = im1[nodeExtent[0, 0, 0]:nodeExtent[0, 1, 0] + 1, \
                        nodeExtent[0, 0, 1]:nodeExtent[0, 1, 1] + 1, \
                        nodeExtent[0, 0, 2]:nodeExtent[0, 1, 2] + 1].copy()

        im2_subvolume = im2[nodeExtent[1, 0, 0]:nodeExtent[1, 1, 0] + 1, \
                        nodeExtent[1, 0, 1]:nodeExtent[1, 1, 1] + 1, \
                        nodeExtent[1, 0, 2]:nodeExtent[1, 1, 2] + 1].copy()

        # make sure that we have enough data to send at least for a correlation window.
        if im1_subvolume.shape != tuple([x * 2 + 1 for x in data.correlation_window]) or im2_subvolume.shape < tuple(
                [x * 2 + 1 for x in data.correlation_window]):
            # workerQueues[workerNumber].put(["Error", None, None])
            dataMessage[0] = "Error"

        else:
            # Reply with data into the worker's data queue
            # workerQueues[workerNumber].put(["Data", im1_subvolume, im2_subvolume])
            dataMessage[0] = "Data"
            dataMessage[1] = im1_subvolume
            dataMessage[2] = im2_subvolume
    except:
        # workerQueues[workerNumber].put(["Error", None, None])
        dataMessage[0] = "Error"

    return dataMessage


def DIC_one_node_map(parameters ):
    nodeNumber = parameters[0]
    extent = parameters[1]
    im1 = parameters[2]
    im2 = parameters[3]
    zExtents_im1_current = parameters[4]
    zExtents_im2_current = parameters[5]
    data = parameters[6]

    return DIC_one_node(nodeNumber, extent, im1, im2, zExtents_im1_current, zExtents_im2_current, data)

def DIC_one_node(nodeNumber,extent, im1, im2,zExtents_im1_current, zExtents_im2_current , data ):


    # while True:
    #     #time.sleep( 1 )
    #     # setupMessage = q_nodes.get()
    #
    #     if setupMessage[0] == "STOP":
    #         try: logging.log.info("DIC_worker %i: Got a request to stop, quitting."%( workerNumber ))
    #         except: "DIC_worker %i: Got a request to stop, quitting."%( workerNumber )
    #         return -1
    #
    #     else:
    # Treat this node...
    # nodeNumber  = setupMessage[0]
    # extent      = setupMessage[1]

    ############################## read subset volume  ###########################
    dataMessage  = read_sub_volume(extent, im1, im2, zExtents_im1_current, zExtents_im2_current, data)


    # Define these high up to be able to return them, even if everything goes wrong.
    error   = 0
    cc      = 0.0
    nodeDisplacement = numpy.array( [ 0.0, 0.0, 0.0 ] )
    nodeDispSubpixel = numpy.array( [ 0.0, 0.0, 0.0 ] )
    nodeRotSubpixel  = numpy.array( [ 0.0, 0.0, 0.0 ] )

    if  dataMessage[0] == "Error":
        # Error fetching the data from the data_delivery_worker, return nothing but a data error (#1)
        error += 1

    elif dataMessage[0] == "Data":
        # Reinitialise all important variables
        im1     = dataMessage[1]
        im1Dim  = im1.shape
        im2     = dataMessage[2]
        im2Dim  = im2.shape

        # Get images out of the reply from data_delivery_worker
        #   NOTE: im2 should be bigger than im1 if the search range is not zero. (it is checked in data_delivery_worker)

        # 2015-01-19 EA: Check that the mean value of the reference im1 is greater than the grey_threshold
        if im1.mean() < data.grey_threshold[0] or im1.mean() > data.grey_threshold[1]:
            # outside the range of interesting gray values, don't correlate, return error = 2
            error += 2

        else:
            # we're within grey threshold... continue...

            # --- Run C-code PIXEL SEARCH ---
            # get displacement of this node from returns...

            returns = pixel_search.pixel_search( im1, im2, 4 )
            # -------------------------------

            nodeDisplacement = returns[0:3]
            cc               = returns[3]


            # We're doing a subpixel search -- either CC or Image Interpolation!
            if any( data.subpixel_mode ):

                # ==================================
                # === data check for CC and II-T ===
                # ==================================

                # Both CC and translation-only image interpolation take im2 = im1+-1 pixel, so prepare it in case we're in this case:
                if ( data.subpixel_mode[0] ) or ( data.subpixel_mode[1] and not data.subpixel_mode[2] ):
                    cornerOffset = 1
                    im2Pm1 = im2[  int(nodeDisplacement[0])-cornerOffset:int(nodeDisplacement[0])+im1Dim[0]+cornerOffset,\
                                   int(nodeDisplacement[1])-cornerOffset:int(nodeDisplacement[1])+im1Dim[1]+cornerOffset,\
                                   int(nodeDisplacement[2])-cornerOffset:int(nodeDisplacement[2])+im1Dim[2]+cornerOffset  ]

                    # === Step 1: Measure the dimensions of image 1 ===
                    im2Pm1Dim = numpy.array( im2Pm1.shape )
                    # 2015-12-17 EA: adding Check for 2D images, the z-dimension will always be 1...
                    if im1.shape[0] == 1 and im2.shape[0] == 1:
                        # then we're dealing with a 2D image
                        if ( im2Pm1Dim[1] - im1Dim[1] ) != 2*cornerOffset or  ( im2Pm1Dim[2] - im1Dim[2] ) != 2*cornerOffset:
                            # We don't have enough data (i.e. +- 1 px) to do a CC interpolation, quit.
                            error           += 32

                    else:
                        # We're in 3D
                        if not all( ( im2Pm1Dim - im1Dim ) == 2*cornerOffset ):
                            # We don't have enough data (i.e. +- 1 px) to do a CC interpolation, quit.
                            error           += 32

                # ==================================


                # ==================================
                # === data check for II-Rotation ===
                # ==================================
                # Also check the data extents for the rotation...
                if ( data.subpixel_mode[1] and data.subpixel_mode[2] ):
                    cornerOffsetRot = int( ( ( numpy.sqrt(3) * max( im1Dim ) ) - max( im1Dim ) + 1 ) / 2.0 )

                    im2Rot = im2[  nodeDisplacement[0]-cornerOffsetRot:nodeDisplacement[0]+im1Dim[0]+cornerOffsetRot,\
                                   nodeDisplacement[1]-cornerOffsetRot:nodeDisplacement[1]+im1Dim[1]+cornerOffsetRot,\
                                   nodeDisplacement[2]-cornerOffsetRot:nodeDisplacement[2]+im1Dim[2]+cornerOffsetRot  ]

                    # === Step 1: Measure the dimensions of image 1 ===
                    im2RotDim = numpy.array( im2Rot.shape )

                    if not all( ( im2RotDim - im1Dim ) == 2*cornerOffsetRot ):
                        # We don't have enough data (i.e. +- 1 px) to do a CC interpolation. asking for more data

                        newExtent = extent.copy()
                        newExtent[1,0] = [  nodeDisplacement[0]-cornerOffsetRot+extent[1,0,0], \
                                            nodeDisplacement[1]-cornerOffsetRot+extent[1,0,1], \
                                            nodeDisplacement[2]-cornerOffsetRot+extent[1,0,2] ]

                        newExtent[1,1] = [  nodeDisplacement[0]+cornerOffsetRot+extent[1,0,0]+im1Dim[0], \
                                            nodeDisplacement[1]+cornerOffsetRot+extent[1,0,1]+im1Dim[1], \
                                            nodeDisplacement[2]+cornerOffsetRot+extent[1,0,2]+im1Dim[2] ]

                        # Make a data request to data_delivery_worker, just with worker number data extents
                        # q_data_requests.put( [ "DataRequest", workerNumber, newExtent ] )
                        # dataMessage = q_data.get()
                        dataMessage = read_sub_volume(newExtent, im1, im2, zExtents_im1_current, zExtents_im2_current, data)

                        if  dataMessage[0] == "Error":
                            # Error fetching the data from the data_delivery_worker, return nothing but a data error (#1)
                            error += 1

                        elif dataMessage[0] == "Data":
                            im2Rot     = dataMessage[2]
                            im2RotDim  = numpy.array( im2Rot.shape )

                            if not all( ( im2RotDim - im1Dim ) == 2*cornerOffsetRot ):
                                #Got more data but it was the wrong shape
                                error           += 32

                # ==================================

                # ===========================
                # === CC INTERPOLATION ======
                # ===========================
                # OK, let's do the CC interpolation if we've been asked to do it!
                if data.subpixel_mode[0] and error == 0:

                    if im1.shape[0] == 1 and im2.shape[0] == 1:
                        returns = cc_interpolation_local_2D(  im1, im2Pm1, \
                                                              data.subpixel_CC_refinement_step_threshold, \
                                                              data.subpixel_CC_max_refinement_iterations, \
                                                              data.subpixel_CC_max_refinement_step  )
                    else:
                        returns = cc_interpolation_local(     im1, im2Pm1, \
                                                              data.subpixel_CC_refinement_step_threshold, \
                                                              data.subpixel_CC_max_refinement_iterations, \
                                                              data.subpixel_CC_max_refinement_step  )

                    nodeDispSubpixel = returns[0:3]
                    ccSubpixel       = returns[3]
                    iterations       = returns[4]
                    error           += returns[5]

                    if ccSubpixel >= cc and error == 0: cc = ccSubpixel
                    else:               error += 1024
                # ===========================


                # ===========================
                # === IMAGE INTERPOLATION ===
                # ===========================
                if data.subpixel_mode[1] and error == 0:
                    if data.subpixel_mode[2]:
                        #Doing Image Interpolation with Translation AND Rotation!
                        guess = numpy.hstack( ( nodeDispSubpixel, nodeRotSubpixel) )

                        returns = image_interpolation_translation_rotation( im1, im2Rot, guess, cornerOffsetRot, data.subpixel_II_interpolationMode, data.subpixel_II_interpolation_order, data.subpixel_II_optimisation_mode )

                        nodeDispSubpixel = returns[0][0:3]
                        nodeRotSubpixel  = returns[0][3:6]
                        ccSubpixel       = returns[1]
                        iterations       = returns[2]
                        error           += returns[3]

                        cc = ccSubpixel

                    else:
                        #Doing Image Interpolation with Translation!
                        returns = image_interpolation_translation_rotation( im1, im2Pm1, nodeDispSubpixel, cornerOffset, data.subpixel_II_interpolationMode, data.subpixel_II_interpolation_order, data.subpixel_II_optimisation_mode )

                        nodeDispSubpixel = returns[0]
                        ccSubpixel       = returns[1]
                        iterations       = returns[2]
                        error           += returns[3]

                        cc = ccSubpixel
                # ===========================

    # In any case send something on the q_results
    # q_results.put( [ nodeNumber, nodeDispSubpixel + nodeDisplacement, nodeRotSubpixel, cc, error ]  )
    return nodeNumber, nodeDispSubpixel + nodeDisplacement, nodeRotSubpixel, cc, error


# ===========================
# === Program Starts Here ===
# ===========================
# @profile
def DIC_parallel( kinematics, data ):

    # Wake up the data_delivery_worker with a new "data" array.
    # q_data_requests.put( [ "NewData", data ] )



    # --- Define the extent of data we need to ask for ---
    # ----- i.e., generate extents matrix ----------------
    # 2014-10-04 EA and ET: updating extents matrix to a 4D array, with { node number }, { im_number 0,1 }, { top, bottom }, { z, y, x }
    extents = numpy.zeros( ( kinematics.shape[0], 2, 2, 3 ), dtype=numpy.int16 )    # change int32 to int16, to save memory (the extent would not be too large)

    print ("finished creating extents array for all nodes, memory usage in bytes, GB, process id" ,process.memory_info().rss,process.memory_info().rss/(1024*1024*1024.0), process.pid)

    #                     position            correlation_window            top extent of search window     prior displacement
    # --- Handling im1_lo
    extents[:,0,0] = kinematics[:,1:4] - data.correlation_window
    # --- Handling im1_hi
    extents[:,0,1] = kinematics[:,1:4] + data.correlation_window
    # --- Handling im2_lo
    extents[:,1,0] = kinematics[:,1:4] - data.correlation_window + numpy.array( data.search_window )[:,0] +  kinematics[:,4:7]
    # --- Handling im2_hi
    extents[:,1,1] = kinematics[:,1:4] + data.correlation_window + numpy.array( data.search_window )[:,1] +  kinematics[:,4:7]
    # ----------------------------------------------------

    # Extents can not exceed the image_slices_extents
    extents[:,:,0,0] = numpy.maximum( extents[:,:,0,0], numpy.ones_like(extents[:,:,0,0]) * data.image_slices_extent[:,0] )
    extents[:,:,1,0] = numpy.minimum( extents[:,:,1,0], numpy.ones_like(extents[:,:,1,0]) * data.image_slices_extent[:,1] )


    # --- Set Up Queues ---
    # This queue will contain the nodes for the DIC workers to process
    # q_nodes         = multiprocessing.Queue()

    # This queue will contain the results for each node
    # q_results       = multiprocessing.Queue( )

    # ----------------------------------

    # Calculate the highest slice we need
    currentTopSlice_image1 = min(extents[:,0,0,0])
    currentTopSlice_image2 = max( min( min(extents[:,1,0,0] ) - int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), data.image_slices_extent[1,1]), 0 )

    # Current bottom slice is the minimum between the lowest slice given by the max on the extents
    #   and the current bottom slice due to memory limit
    currentBottomSlice_image1 = min( max(extents[:,0,1,0]), currentTopSlice_image1 + data.memLimitSlices)
    currentBottomSlice_image2 = min( max(extents[:,1,1,0]) + int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), currentTopSlice_image2 + data.memLimitSlices)

    # Initializing a node done table
    # nodeDoneTable = numpy.zeros( ( kinematics.shape[0], 1 ), dtype = bool )

    # Make sure no node data extents are larger than the memory limit in the vertical direction
    # extentsCheck = (extents[:,:,1,0] - extents[:,:,0,0]) > data.memLimitSlices

    # if extentsCheck.any() :
    #   # try: logging.err.error("DIC_setup(): The memory limit set does not fulfil the required vertical extents for at least one node")
    #   # except: print "DIC_setup(): The memory limit set does not fulfil the required vertical extents for at least one node"
    #   # for workerNumber in range( data.nWorkers ):
    #   #     q_nodes.put( [ "STOP" ] )
    #   print ("DIC_setup(): The memory limit set does not fulfil the required vertical extents for at least one node")
    #   return

    # If this is the case mark this node as done so it is not processed and give an error
    # nodeDoneTable[ numpy.logical_or( extentsCheck[:,0], extentsCheck[:,1] ) ] = True
    # kinematics[    numpy.logical_or( extentsCheck[:,0], extentsCheck[:,1] ), 11 ] += 512

    # --- Variables for receive queue management ---
    # nNodes_to_correlate = kinematics[ :, 0 ].shape[0] - sum( nodeDoneTable )
    # nodesProcessedTotal  = 0
    # printInterval   = max( 1, int(nNodes_to_correlate*data.printPercent/100.0) )
    # ----------------------------------------------

    # --- These two variables are to have a calculation time ---
    calculationTimeA = time.time()
    # prevNodesProcessed = 0
    # ----------------------------------------------------------

    #################### loading images  ####################
    # Initialise current z-extents and empty images
    zExtents_im1_current = [ -1, -1 ]
    zExtents_im2_current = [ -1, -1 ]
    # initialise empty initial images
    im1 = None
    im2 = None

    zExtents_im1_new = [ currentTopSlice_image1, currentBottomSlice_image1 ]
    zExtents_im2_new = [ currentTopSlice_image2, currentBottomSlice_image2 ]

    try:
        # load new slices, if any, and update current z extents.
        try:
            logging.log.info("data_delivery_worker(): Loading data...")
        except:
            print  "data_delivery_worker(): Loading data..."
        zExtents_im1_current, im1 = load_slices(zExtents_im1_new, zExtents_im1_current, im1, 1, data)
        zExtents_im2_current, im2 = load_slices(zExtents_im2_new, zExtents_im2_current, im2, 2, data)
        try:
            logging.log.info("data_delivery_worker(): Done")
        except:
            print  "data_delivery_worker(): Done"
    except Exception as exc:
        # raise Exception(exc)
        try:
            logging.err.error(exc.message)
        except:
            print exc.message
        im1 = []
        im2 = []


    # parallel run the image correlation
    num_cores = multiprocessing.cpu_count()
    print('number of thread %d' % num_cores)
    # theadPool = mp.Pool(num_cores)  # multi threads, can not utilize all the CPUs? not sure hlc 2018-4-19
    theadPool = Pool(num_cores)  # multi processes
    parameters_list = []
    for nodeNumber in range(kinematics.shape[0]):
        nodeExtent = extents[nodeNumber]
        if nodeExtent[0, 0, 0] >= currentTopSlice_image1 \
                and nodeExtent[0, 1, 0] <= currentBottomSlice_image1 \
                and nodeExtent[1, 0, 0] >= currentTopSlice_image2 \
                and nodeExtent[1, 1, 0] <= currentBottomSlice_image2:
            parameters_list.append([nodeNumber,nodeExtent, im1, im2,zExtents_im1_current, zExtents_im2_current ,data])

    results = theadPool.map(DIC_one_node_map, parameters_list)


    # #  for test serial (not parallel)
    # results = []
    # count = 0
    # for nodeNumber in range(kinematics.shape[0]):
    #     nodeExtent = extents[nodeNumber]
    #     if nodeExtent[0, 0, 0] >= currentTopSlice_image1 \
    #             and nodeExtent[0, 1, 0] <= currentBottomSlice_image1 \
    #             and nodeExtent[1, 0, 0] >= currentTopSlice_image2 \
    #             and nodeExtent[1, 1, 0] <= currentBottomSlice_image2:
    #         res = DIC_one_node(nodeNumber,nodeExtent, im1, im2,zExtents_im1_current, zExtents_im2_current ,data)
    #         results.append(res)
    #         count += 1
    #         print("completed: %d /%d"%(count,kinematics.shape[0]))



    # get results
    for message in results:

        # Extract result of this thread's result
        nodeNumber = message[0]

        #  Since C-code pixel search doesn't know about the prior field and the search window, add these back in now, in order for the
        #   displacement to be absolute.
        message[1][0] += extents[nodeNumber,1,0,0] - extents[nodeNumber,0,0,0]
        message[1][1] += extents[nodeNumber,1,0,1] - extents[nodeNumber,0,0,1]
        message[1][2] += extents[nodeNumber,1,0,2] - extents[nodeNumber,0,0,2]

        # Copy into relevant results matrices...
        kinematics[ nodeNumber, 4  ]  = message[1][0]
        kinematics[ nodeNumber, 5  ]  = message[1][1]
        kinematics[ nodeNumber, 6  ]  = message[1][2]
        kinematics[ nodeNumber, 7  ]  = message[2][0]
        kinematics[ nodeNumber, 8  ]  = message[2][1]
        kinematics[ nodeNumber, 9  ]  = message[2][2]
        kinematics[ nodeNumber, 10 ]  = message[3]
        kinematics[ nodeNumber, 11 ] += message[4]     # Error is additive


    # #Outside loop continues until there are no nodes left
    # while not nodeDoneTable.all():
    #
    #     # --- update NewExtents for the data_delivery_worker for the newly added nodes ---
    #     # q_data_requests.put( [ "NewExtents", [ [ currentTopSlice_image1, currentBottomSlice_image1 ], [ currentTopSlice_image2, currentBottomSlice_image2 ] ] ] )
    #     # --------------------------------------------------
    #
    #     # reset node counter -- in order to know when to stop...
    #     # nodesToProcess = 0
    #
    #     # For every node check if it has been done and if not whether it is inside the current block of data
    #     for nodeNumber in range( kinematics.shape[0] ):
    #         nodeExtent = extents[ nodeNumber ]
    #         if not nodeDoneTable[ nodeNumber ]  and nodeExtent[0,0,0] >= currentTopSlice_image1    \
    #                                             and nodeExtent[0,1,0] <= currentBottomSlice_image1 \
    #                                             and nodeExtent[1,0,0] >= currentTopSlice_image2    \
    #                                             and nodeExtent[1,1,0] <= currentBottomSlice_image2 :
    #
    #             # Adding the node to the queue for the DIC_worker and update nodeDoneTable
    #             q_nodes.put( [ nodeNumber, nodeExtent ] )
    #             nodeDoneTable[ nodeNumber ] = True
    #             # Add one to node counter...
    #             nodesToProcess += 1
    #
    #     # Checking if all nodes have been sent to the worker add STOP to queue to stop the DIC_workers
    #     if nodeDoneTable.all():
    #         for workerNumber in range( data.nWorkers ):
    #             q_nodes.put( [ "STOP" ] )
    #
    #     # Updating current slices
    #     # Calculate the highest slice from NOT done nodes
    #     if ( nodeDoneTable == False ).any():
    #         currentTopSlice_image1 = min( extents[ numpy.where( nodeDoneTable == False )[0], 0, 0, 0] )
    #         currentTopSlice_image2 = max( min( min( extents[ numpy.where( nodeDoneTable == False )[0], 1, 0, 0] - int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ) ), data.image_slices_extent[1,1]), 0 )
    #
    #     # Current bottom slice is the minimum between the lowest slice given by the max on the extents
    #     #   and the current bottom slice due to memory limit
    #     currentBottomSlice_image1 = min( max(extents[:,0,1,0]), currentTopSlice_image1 + data.memLimitSlices)
    #     currentBottomSlice_image2 = min( max(extents[:,1,1,0]) + int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), currentTopSlice_image2 + data.memLimitSlices)
    #
    #
    #     # Loop until all workers have hanged up
    #     #while finishedThreads < data.nWorkers:
    #     while nodesToProcess > 0:
    #
    #         message = q_results.get()
    #
    #         nodesProcessedTotal  += 1
    #         nodesToProcess       -= 1
    #
    #         if nodesProcessedTotal%printInterval == 0:
    #
    #               print "\r\tCompleted node number %05i  ( %2.2f %% )"%( nodesProcessedTotal, 100*(nodesProcessedTotal)/float(nNodes_to_correlate) ),
    #
    #               # --- Calculation of remaining time ---
    #               calculationTimeB = time.time()
    #               nodesProcessedTotalThisStep = nodesProcessedTotal - prevNodesProcessed
    #               secondsForThisStep      = calculationTimeB - calculationTimeA
    #               try:
    #                 nodesPerSecond         = nodesProcessedTotalThisStep / float( secondsForThisStep )
    #               except ZeroDivisionError:
    #                 nodesPerSecond         = nodesProcessedTotalThisStep
    #               nodesRemaining          = nNodes_to_correlate - nodesProcessedTotal
    #               secondsRemaining        = ( nodesRemaining / float( nodesPerSecond ) )[0]
    #
    #
    #               # 2014-10-03 EA: Estimating computation time between printouts...
    #               print "\tTime remaining = ~%s\033[K"%( display_time( secondsRemaining ) ),
    #               sys.stdout.flush()
    #
    #               # update counters:
    #               calculationTimeA       = time.time()
    #               prevNodesProcessed     = nodesProcessedTotal
    #               # -------------------------------------
    #
    #         # Extract result of this thread's result
    #         nodeNumber = message[0]
    #
    #         #  Since C-code pixel search doesn't know about the prior field and the search window, add these back in now, in order for the
    #         #   displacement to be absolute.
    #         message[1][0] += extents[nodeNumber,1,0,0] - extents[nodeNumber,0,0,0]
    #         message[1][1] += extents[nodeNumber,1,0,1] - extents[nodeNumber,0,0,1]
    #         message[1][2] += extents[nodeNumber,1,0,2] - extents[nodeNumber,0,0,2]
    #
    #         # Copy into relevant results matrices...
    #         kinematics[ nodeNumber, 4  ]  = message[1][0]
    #         kinematics[ nodeNumber, 5  ]  = message[1][1]
    #         kinematics[ nodeNumber, 6  ]  = message[1][2]
    #         kinematics[ nodeNumber, 7  ]  = message[2][0]
    #         kinematics[ nodeNumber, 8  ]  = message[2][1]
    #         kinematics[ nodeNumber, 9  ]  = message[2][2]
    #         kinematics[ nodeNumber, 10 ]  = message[3]
    #         kinematics[ nodeNumber, 11 ] += message[4]     # Error is additive
    #
    #         # 2015-07-30 - EA: cc_percent badly applied, applying it differently, this is not elegant, but at least not wrong:
    #         if data.cc_percent:
    #           kinematics[ nodeNumber, 10 ]  = kinematics[ nodeNumber, 10 ] * 100

    return kinematics
