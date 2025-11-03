all:
	nvcc -std=c++11 -gencode arch=compute_120,code=sm_120 -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w
# -DMEMORY_CHECK 		# meory check
# -DCONTROL_MATCH		# control match
# -DDEBUG 				# debug
# -DTIMER 				# timer
# --ptxas-options=-v	# print ptxas information

# -DFIGURE9_SUM			# coarsen adjwgtsum
# -DFIGURE9_TIME		# coarsen adjwgtsum time
# -DFIGURE10_CGRAPH		# init graph
# -DFIGURE10_EXHAUSTIVE	# init exhaustive edgecut
# -DFIGURE10_SAMPLING	# init sampling edgecut
# -DFIGURE14_EDGECUT	# final edgecut