#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail

BIOSPPY_WHL="biosppy-0.3.0-py2.py3-none-any.whl"
PYENTRP="pyentrp-0.3.1.tar.gz"
PYWAVELETS="PyWavelets-0.5.2.tar.gz"
BASEDIR=$(dirname "$0")

#Install BioSPPY Library
pip3 install --user $BASEDIR/$BIOSPPY_WHL

#Install PyEntropy
pip3 install --user $BASEDIR/$PYENTRP

#Install PyWavelets
pip3 install --user $BASEDIR/$PYWAVELETS

#Install PyEEG
python3 setup-pyeeg.py install --user