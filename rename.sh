#!/bin/bash


set -e

for lsf in $(find . -maxdepth 1 -type f -name 'lsf.o*'); do
    dataset=$(head -n 3 "$lsf" | grep -Eo 'dataset=[^ ]*' | cut -d '=' -f 2)
    model=$(head -n 3 "$lsf" | grep -Eo 'model=[^ ]*' | cut -d '=' -f 2)
    fixed=$(head -n 3 "$lsf" | grep -Eo 'fixed_curvature=[^ ]*' | cut -d '=' -f 2)
    suffix=$(basename $lsf)
    mv -v $lsf $dataset.fixed${fixed}.$model.$suffix
done
