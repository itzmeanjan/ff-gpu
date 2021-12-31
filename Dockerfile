FROM intel/oneapi-basekit
RUN apt-get update; apt-get install -y git
WORKDIR /home
RUN git clone https://github.com/itzmeanjan/ff-gpu.git; cd ff-gpu; sed -i 's/clang++/dpcpp/g' Makefile; cd ..; echo "cd ff-gpu; make test; make clean" > run_test
CMD ["bash", "run_test"]
