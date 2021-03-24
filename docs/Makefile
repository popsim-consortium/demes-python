
# Need to set PYTHONPATH so that we pick up the local demes
PYPATH=${PWD}/..
DEMES_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import demes; print(demes.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for msprime version ${DEMES_VERSION}
	sed -i s/__DEMES_VERSION__/${DEMES_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
