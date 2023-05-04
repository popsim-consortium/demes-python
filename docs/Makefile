
# Need to set PYTHONPATH so that we pick up the local demes
export PYTHONPATH=$(realpath ..)
DEMES_VERSION:=$(shell PYTHONPATH=${PYTHONPATH} \
	python3 -c 'import demes; print(demes.__version__.split("+")[0])')

dev:
	./build.sh

clean:
	rm -fR _build
