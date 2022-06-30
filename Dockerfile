FROM intel/intel-optimized-pytorch

RUN mkdir /work/
COPY ./ /work/
WORKDIR /work/