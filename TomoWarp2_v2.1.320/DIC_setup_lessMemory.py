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

# Modified from DIC_setup.py by Lingcao Huang, 21 May 2020
# In the original DIC_setup, when the node spacing is very small (such as 1), it will have a lot of nodes, and end in run out of memory errors.


""" 
INPUTS:
  - kinematics
  - "data" structure
  - q_data_requests to comunicate with data_delivery_worker
  - workerQueues list of queues that allows comunication of data from data_delivery_worker 
    to DIC_worker
  
OUTPUTS:
  - filled-in kinematics
"""

import sys, time
import numpy
import multiprocessing
import logging

from DIC_worker import DIC_worker
from tools.print_variable import pv


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
from psutil import virtual_memory
process = psutil.Process(os.getpid())

import gc
# ===========================
# === Program Starts Here ===
# ===========================
# @profile
def DIC_setup_lessMemory( kinematics, data, q_data_requests, workerQueues ):

    # Wake up the data_delivery_worker with a new "data" array.
    q_data_requests.put( [ "NewData", data ] )
    logging.log.info("The number of nodes for this subset: %d"%kinematics.shape[0])

    # --- Define the extent of data we need to ask for ---
    # ----- i.e., generate extents matrix ----------------
    # 2014-10-04 EA and ET: updating extents matrix to a 4D array, with { node number }, { im_number 0,1 }, { top, bottom }, { z, y, x }
    extents = numpy.zeros( ( kinematics.shape[0], 2, 2, 3 ), dtype=numpy.int )    # change int32 to int16, to save memory (the extent would not be too large)

    logging.log.info ("finished creating extents array for all nodes, memory usage in bytes, GB, process id" ,process.memory_info().rss,process.memory_info().rss/(1024*1024*1024.0), process.pid)

    #                     position            correlation_window            top extent of search window     prior displacement
    # --- Handling im1_lo
    extents[:,0,0] = kinematics[:,1:4] - data.correlation_window
    gc.collect()
    # --- Handling im1_hi
    extents[:,0,1] = kinematics[:,1:4] + data.correlation_window
    gc.collect()
    # --- Handling im2_lo
    extents[:,1,0] = kinematics[:,1:4] - data.correlation_window + numpy.array( data.search_window )[:,0] +  kinematics[:,4:7]
    gc.collect()
    # --- Handling im2_hi
    extents[:,1,1] = kinematics[:,1:4] + data.correlation_window + numpy.array( data.search_window )[:,1] +  kinematics[:,4:7]
    gc.collect()
    # ----------------------------------------------------

    # Extents can not exceed the image_slices_extents
    extents[:,:,0,0] = numpy.maximum( extents[:,:,0,0], numpy.ones_like(extents[:,:,0,0]) * data.image_slices_extent[:,0] )
    gc.collect()
    extents[:,:,1,0] = numpy.minimum( extents[:,:,1,0], numpy.ones_like(extents[:,:,1,0]) * data.image_slices_extent[:,1] )
    gc.collect()

    logging.log.info ("finished extents calculation, memory usage in bytes:%d, GB:%lf, process id:%d"%(process.memory_info().rss,
    process.memory_info().rss / (1024 * 1024 * 1024.0), process.pid))

    # --- Set Up Queues ---
    # This queue will contain the nodes for the DIC workers to process
    q_nodes         = multiprocessing.Queue()

    # This queue will contain the results for each node
    q_results       = multiprocessing.Queue( )

    logging.log.info ("finished Setting Up q_nodes and q_results, memory usage in bytes: %d, GB:%lf, process id:%d"% (process.memory_info().rss,
    process.memory_info().rss / (1024 * 1024 * 1024.0), process.pid))

    # --- Launch DIC worker nodes  ---
    for workerNumber in range( data.nWorkers ):
        p = multiprocessing.Process( target=DIC_worker, args=( workerNumber, q_nodes, q_results, q_data_requests, workerQueues[ workerNumber ], data ) )
        p.start()
    # ----------------------------------

    logging.log.info ("finished Launching DIC worker, memory usage in bytes: %d, GB:%lf, process id:%d"% (process.memory_info().rss,
    process.memory_info().rss / (1024 * 1024 * 1024.0), process.pid))

    # Calculate the highest slice we need
    currentTopSlice_image1 = min(extents[:,0,0,0])
    currentTopSlice_image2 = max( min( min(extents[:,1,0,0] ) - int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), data.image_slices_extent[1,1]), 0 )

    # Current bottom slice is the minimum between the lowest slice given by the max on the extents
    #   and the current bottom slice due to memory limit
    currentBottomSlice_image1 = min( max(extents[:,0,1,0]), currentTopSlice_image1 + data.memLimitSlices)
    currentBottomSlice_image2 = min( max(extents[:,1,1,0]) + int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), currentTopSlice_image2 + data.memLimitSlices)

    # Initializing a node done table
    nodeDoneTable = numpy.zeros( ( kinematics.shape[0], 1 ), dtype = bool )

    # Make sure no node data extents are larger than the memory limit in the vertical direction
    extentsCheck = (extents[:,:,1,0] - extents[:,:,0,0]) > data.memLimitSlices

    if extentsCheck.any() :
      try: logging.err.error("DIC_setup(): The memory limit set does not fulfil the required vertical extents for at least one node")
      except: print "DIC_setup(): The memory limit set does not fulfil the required vertical extents for at least one node"
      for workerNumber in range( data.nWorkers ):
          q_nodes.put( [ "STOP" ] )
      return

    # If this is the case mark this node as done so it is not processed and give an error
    #TODO: could conditionally accept these nodes
    nodeDoneTable[ numpy.logical_or( extentsCheck[:,0], extentsCheck[:,1] ) ] = True
    kinematics[    numpy.logical_or( extentsCheck[:,0], extentsCheck[:,1] ), 11 ] += 512

    # --- Variables for receive queue management ---
    nNodes_to_correlate = kinematics[ :, 0 ].shape[0] - sum( nodeDoneTable )
    nodesProcessedTotal  = 0
    printInterval   = max( 1, int(nNodes_to_correlate*data.printPercent/100.0) )
    # ----------------------------------------------

    # --- These two variables are to have a calculation time ---
    calculationTimeA = time.time()
    prevNodesProcessed = 0
    # ----------------------------------------------------------

    max_nodes_in_quene = 5000
    mem = virtual_memory()
    # mem.total  # total physical memory
    # mem.available # total available memory
    # each node, need to copy subvolume of image
    print(data.correlation_window)
    print(data.search_window)
    corr_size = [ win*2 +1  for win in data.correlation_window ]
    print('correlation size',corr_size)
    subVolume1 = corr_size[0]*corr_size[1]*corr_size[2]*4  # bytes
    search_size = [ win*2 + 1 + abs(ed[0]) + abs(ed[1]) for win, ed in zip(data.correlation_window,data.search_window) ]
    subVolume2 = search_size[0]*search_size[1]*search_size[2]*4 # bytes
    print("search_size",search_size)
    bytes_for_received_result = kinematics.shape[0]*9*4 # bytes
    print("available memory (GB), reserved for results: ",mem.available/(1024.0*1024*1024.0),bytes_for_received_result/(1024.0*1024*1024.0))
    max_nodes_in_quene = (mem.available - bytes_for_received_result)/(subVolume1+subVolume2)
    print ("max_nodes_in_quene:",max_nodes_in_quene)
    break_quene_times = 0

    #Outside loop continues until there are no nodes left
    while not nodeDoneTable.all():

        # --- update NewExtents for the data_delivery_worker for the newly added nodes ---
        if break_quene_times == 0:
            q_data_requests.put( [ "NewExtents", [ [ currentTopSlice_image1, currentBottomSlice_image1 ], [ currentTopSlice_image2, currentBottomSlice_image2 ] ] ] )
        # --------------------------------------------------

        # reset node counter -- in order to know when to stop...
        nodesToProcess = 0

        # For every node check if it has been done and if not whether it is inside the current block of data
        for nodeNumber in range( kinematics.shape[0] ):
            nodeExtent = extents[ nodeNumber ]
            if not nodeDoneTable[ nodeNumber ]  and nodeExtent[0,0,0] >= currentTopSlice_image1    \
                                                and nodeExtent[0,1,0] <= currentBottomSlice_image1 \
                                                and nodeExtent[1,0,0] >= currentTopSlice_image2    \
                                                and nodeExtent[1,1,0] <= currentBottomSlice_image2 :

                # Adding the node to the queue for the DIC_worker and update nodeDoneTable
                q_nodes.put( [ nodeNumber, nodeExtent ] )
                nodeDoneTable[ nodeNumber ] = True
                # Add one to node counter...
                nodesToProcess += 1
            if nodesToProcess >  max_nodes_in_quene:
                break_quene_times = break_quene_times + 1
                break

        # Checking if all nodes have been sent to the worker add STOP to queue to stop the DIC_workers
        if nodeDoneTable.all():
            for workerNumber in range( data.nWorkers ):
                q_nodes.put( [ "STOP" ] )

        # Updating current slices
        # Calculate the highest slice from NOT done nodes
        if ( nodeDoneTable == False ).any():
            currentTopSlice_image1 = min( extents[ numpy.where( nodeDoneTable == False )[0], 0, 0, 0] )
            currentTopSlice_image2 = max( min( min( extents[ numpy.where( nodeDoneTable == False )[0], 1, 0, 0] - int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ) ), data.image_slices_extent[1,1]), 0 )

        # Current bottom slice is the minimum between the lowest slice given by the max on the extents
        #   and the current bottom slice due to memory limit
        currentBottomSlice_image1 = min( max(extents[:,0,1,0]), currentTopSlice_image1 + data.memLimitSlices)
        currentBottomSlice_image2 = min( max(extents[:,1,1,0]) + int( data.subpixel_mode[2]*max(data.correlation_window)*numpy.sqrt(3)+1 ), currentTopSlice_image2 + data.memLimitSlices)


        # Loop until all workers have hanged up
        #while finishedThreads < data.nWorkers:
        while nodesToProcess > 0:

            message = q_results.get()

            nodesProcessedTotal  += 1
            nodesToProcess       -= 1
            
            if nodesProcessedTotal%printInterval == 0:
              
                  print "\r\tCompleted node number %05i  ( %2.2f %% )"%( nodesProcessedTotal, 100*(nodesProcessedTotal)/float(nNodes_to_correlate) ),

                  # --- Calculation of remaining time ---
                  calculationTimeB = time.time()
                  nodesProcessedTotalThisStep = nodesProcessedTotal - prevNodesProcessed
                  secondsForThisStep      = calculationTimeB - calculationTimeA
                  try:
                    nodesPerSecond         = nodesProcessedTotalThisStep / float( secondsForThisStep )
                  except ZeroDivisionError:
                    nodesPerSecond         = nodesProcessedTotalThisStep
                  nodesRemaining          = nNodes_to_correlate - nodesProcessedTotal
                  secondsRemaining        = ( nodesRemaining / float( nodesPerSecond ) )[0]


                  # 2014-10-03 EA: Estimating computation time between printouts...
                  print "\tTime remaining = ~%s\033[K"%( display_time( secondsRemaining ) ),
                  sys.stdout.flush()

                  # update counters:
                  calculationTimeA       = time.time()
                  prevNodesProcessed     = nodesProcessedTotal
                  # -------------------------------------

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

            # 2015-07-30 - EA: cc_percent badly applied, applying it differently, this is not elegant, but at least not wrong:
            if data.cc_percent:
              kinematics[ nodeNumber, 10 ]  = kinematics[ nodeNumber, 10 ] * 100

    return kinematics
