#!/bin/bash

# Download data
if [ ! -d $PWD/data ]; then
  TARBALL=data.tar.gz
  echo "Downloading data..."
  wget -O ${TARBALL} -q https://surfdrive.surf.nl/files/index.php/s/NEMstJ83kNRhGwf/download
  tar -xzf ${TARBALL}
  rm ${TARBALL}
  echo "Done."
else
  echo "Already downloaded"
fi
