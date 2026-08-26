# Build the golden harnesses around CMAQ's PPM kernels.
#
# The kernels in fortran/ are compiled UNMODIFIED. Everything CMAQ-specific
# they need is supplied by harness/stubs.f90 or by the cpp defines below.
#
#   make          build both harnesses into build/
#   make clean    remove build/

# `?=` is not enough here: make has a built-in default FC = f77, which counts
# as defined. Only override when the value is still that built-in default, so
# an explicit `make FC=ifort` from the environment still wins.
ifeq ($(origin FC),default)
FC := gfortran
endif
BUILD   := build
SRC     := fortran
HARNESS := harness

# -ffixed-line-length-132: CMAQ fixed-form sources run past column 72.
# -cpp: the kernels contain #ifdef blocks.
# We build WITHOUT -Dparallel, -Disam and -Dsens, which selects the plain
# serial no-instrumentation path through both kernels.
#
# SUBST_HI_LO_BND_PE is a cpp macro in CMAQ's build (bldit_cctm.csh:413),
# expanding to SE_* or NOOP_* depending on ParOpt. We point it at our stub.
FFLAGS  ?= -O2 -g -std=legacy -ffixed-line-length-132 -cpp \
           -DSUBST_HI_LO_BND_PE=stub_hi_lo_bnd_pe \
           -J$(BUILD)

# Free-form for our own .f90 sources; the -ffixed-line-length flag is harmless
# there because gfortran picks form from the extension.

HARNESSES := $(BUILD)/harness_hppm $(BUILD)/harness_vppm $(BUILD)/harness_ppm_coeffs \
             $(BUILD)/harness_zfdbc

.PHONY: all clean
all: $(HARNESSES)

$(BUILD):
	mkdir -p $(BUILD)

# Stub modules must exist before anything that USEs them.
$(BUILD)/stubs.o: $(HARNESS)/stubs.f90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/hppm.o: $(SRC)/hppm.F $(BUILD)/stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/vppm.o: $(SRC)/vppm.F $(BUILD)/stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_hppm.o: $(HARNESS)/harness_hppm.f90 $(BUILD)/stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_vppm.o: $(HARNESS)/harness_vppm.f90 $(BUILD)/stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_hppm: $(BUILD)/stubs.o $(BUILD)/hppm.o $(BUILD)/harness_hppm.o
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/harness_vppm: $(BUILD)/stubs.o $(BUILD)/vppm.o $(BUILD)/harness_vppm.o
	$(FC) $(FFLAGS) $^ -o $@

# PPM lives in vppm.F alongside VPPM, so this links the same object.
$(BUILD)/harness_ppm_coeffs.o: $(HARNESS)/harness_ppm_coeffs.f90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_ppm_coeffs: $(BUILD)/stubs.o $(BUILD)/vppm.o $(BUILD)/harness_ppm_coeffs.o
	$(FC) $(FFLAGS) $^ -o $@

# zfdbc.f is lowercase-.f fixed form and needs no preprocessing, but the same
# flags are harmless and keep one rule for every Fortran source here.
$(BUILD)/zfdbc.o: $(SRC)/zfdbc.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_zfdbc.o: $(HARNESS)/harness_zfdbc.f90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_zfdbc: $(BUILD)/zfdbc.o $(BUILD)/harness_zfdbc.o
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)
