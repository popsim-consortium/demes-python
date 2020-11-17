
# Need to set PYTHONPATH so that we pick up the local demes
export PYTHONPATH=$(realpath ..)
DEMES_VERSION:=$(shell PYTHONPATH=${PYTHONPATH} \
	python3 -c 'import demes; print(demes.__version__.split("+")[0])')

dev:
	./build.sh

dist:
	@echo Building distribution for msprime version ${DEMES_VERSION}
	sed -i s/__DEMES_VERSION__/${DEMES_VERSION}/g _config.yml
	./build.sh

clean:
	rm -fR _build
