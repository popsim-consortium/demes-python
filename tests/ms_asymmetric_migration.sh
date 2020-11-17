#!/bin/sh
#
# From the ms manual:
#
#	M[i][j] is the fraction of subpopulation i which is made up of
#	migrants from subpopulation j each generation.
#
# But migration matrices are an ongoing source of confusion, and its not 100%
# clear from the docs if this really is to be interpreted as forwards-in-time
# movement of migrant individuals, or if this is actually talking about the
# backwards-in-time movement of lineages in the coalescent process.
#
# To confirm that ms really is using the plain old forwards-in-time migration
# language, we construct a three deme model with asymmetric migration.
#
# |         |
# |____     |
# |    | <- |
# |    | <- |
# |    | <- |
# 1    2    3
#
# We choose a very ancient split time for demes 1 and 2,
# and a very high migration rate between demes 2 and 3.
# If we run a coalescent simulation to draw one sample from each of
# deme 1 and deme 3, then the two samples will coalesce
# (with high probability), and the exit code is zero.
# However, if we reverse the migration indices then we observe the error:
#
# 	Infinite coalescent time. No migration.
#
# Which is accompanied by a non-zero exit code.
# In both cases the behaviour is the same with msprime's mspms CLI.

MS=ms
#MS=mspms

# Set M[3][2] = 1.0 using the -em option
# Coalescence expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-em 0.0 3 2 1.0 \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? != "0" ]; then
	echo "samples should have coalesced, but didn't"
	exit 1
fi

# Set M[3][2] = 1.0 using the -ema option
# Coalescence expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-ema 0.0 3 x 0 0 0 x 0 0 1.0 x \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? != "0" ]; then
	echo "samples should have coalesced, but didn't"
	exit 1
fi

# Set M[3][2] = 1.0 using the -ma option
# Coalescence expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-ma x 0 0 0 x 0 0 1.0 x \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? != "0" ]; then
	echo "samples should have coalesced, but didn't"
	exit 1
fi

# Set M[3][2] = 1.0 using the -m option
# Coalescence expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-m 3 2 1.0 \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? != "0" ]; then
	echo "samples should have coalesced, but didn't"
	exit 1
fi

# Set M[2][3] = 1.0 using the -em option
# Infinite waiting time expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-em 0.0 2 3 1.0 \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? = "0" ]; then
	echo "samples coalesced, but shouldn't have"
	exit 1
fi

# Set M[2][3] = 1.0 using the -ema option
# Infinite waiting time expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-ema 0.0 3 x 0 0 0 x 1.0 0 0 x \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? = "0" ]; then
	echo "samples coalesced, but shouldn't have"
	exit 1
fi

# Set M[2][3] = 1.0 using the -ma option
# Infinite waiting time expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-ma x 0 0 0 x 1.0 0 0 x \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? = "0" ]; then
	echo "samples coalesced, but shouldn't have"
	exit 1
fi

# Set M[2][3] = 1.0 using the -m option
# Infinite waiting time expected.
$MS 2 1000 \
	-t 1.0 \
	-I 3 1 0 1 \
	-m 2 3 1.0 \
	-ej 100.0 2 1 \
	-eM 100.0 0.0 \
	>/dev/null 2>&1
if [ $? = "0" ]; then
	echo "samples coalesced, but shouldn't have"
	exit 1
fi
