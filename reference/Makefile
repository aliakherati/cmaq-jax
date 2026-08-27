# Build the golden harnesses around CMAQ's advection code.
#
# Everything under fortran/ and stenex/ is compiled UNMODIFIED. What CMAQ
# normally supplies through its own modules comes instead from harness/stubs*.f90
# and the cpp defines below, which mirror a serial (non-MPI) CMAQ build.
#
#   make          build every harness into build/
#   make clean    remove build/

# `?=` is not enough here: make has a built-in default FC = f77, which counts
# as defined. Only override when the value is still that built-in default, so
# an explicit `make FC=ifort` from the environment still wins.
ifeq ($(origin FC),default)
FC := gfortran
endif
BUILD   := build
SRC     := fortran
STENEX  := stenex
HARNESS := harness
INCLUDE := include

# The SUBST_* names are cpp macros in CMAQ's build (bldit_cctm.csh:400-420).
# A serial build points them at the no-op stencil-exchange layer, which is
# exactly what stenex/ holds, so we use the same mapping rather than inventing
# a stub for it.
STENEX_DEFS := \
  -DSUBST_BARRIER=noop_barrier \
  -DSUBST_GLOBAL_MAX=noop_global_max \
  -DSUBST_GLOBAL_MIN=noop_global_min \
  -DSUBST_GLOBAL_MIN_DATA=noop_global_min_data \
  -DSUBST_GLOBAL_TO_LOCAL_COORD=noop_global_to_local_coord \
  -DSUBST_GLOBAL_SUM=noop_global_sum \
  -DSUBST_GLOBAL_LOGICAL=noop_global_logical \
  -DSUBST_GLOBAL_GATHER=noop_global_gather \
  -DSUBST_GLOBAL_BCAST=noop_global_bcast \
  -DSUBST_LOOP_INDEX=noop_loop_index \
  -DSUBST_SUBGRID_INDEX=noop_subgrid_index \
  -DSUBST_HI_LO_BND_PE=noop_hi_lo_bnd_pe \
  -DSUBST_SUM_CHK=noop_sum_chk \
  -DSUBST_SE_INIT=noop_init \
  -DSUBST_INIT_ARRAY=noop_init_array \
  -DSUBST_COMM=noop_comm \
  -DSUBST_MY_REGION=noop_my_region \
  -DSUBST_SLICE=noop_slice \
  -DSUBST_GATHER=noop_gather \
  -DSUBST_DATA_COPY=noop_data_copy \
  -DSUBST_IN_SYN=noop_in_syn

# INCLUDE SUBST_CONST and friends are cpp macros naming CMAQ include files,
# vendored under include/ (bldit_cctm.csh:506-508).
INCLUDE_DEFS := \
  -DSUBST_CONST=\"CONST.EXT\" \
  -DSUBST_FILES_ID=\"FILES_CTM.EXT\" \
  -DSUBST_PE_COMM=\"PE_COMM.EXT\"

# -ffixed-line-length-132: CMAQ fixed-form sources run past column 72.
# -cpp: the sources contain #ifdef blocks and the macros above.
# Built WITHOUT -Dparallel, -Disam and -Dsens, which selects the plain serial
# no-instrumentation path through every routine.
FFLAGS  ?= -O2 -g -std=legacy -ffixed-line-length-132 -cpp \
           $(STENEX_DEFS) $(INCLUDE_DEFS) \
           -I$(INCLUDE) -I$(BUILD) -J$(BUILD)

STENEX_SRC := $(wildcard $(STENEX)/noop_*.f)
STENEX_OBJ := $(patsubst $(STENEX)/%.f,$(BUILD)/%.o,$(STENEX_SRC))

# Modules every harness links against.
COMMON_OBJ := $(BUILD)/stubs.o $(BUILD)/stubs_driver.o $(STENEX_OBJ)

HARNESSES := $(BUILD)/harness_hppm $(BUILD)/harness_vppm $(BUILD)/harness_ppm_coeffs \
             $(BUILD)/harness_zfdbc $(BUILD)/harness_hadv $(BUILD)/harness_zadv \
             $(BUILD)/harness_hcdiff3d $(BUILD)/harness_hdiff \
             $(BUILD)/harness_tri $(BUILD)/harness_matrix1 $(BUILD)/harness_eddyx \
             $(BUILD)/harness_vdiff

.PHONY: all clean
all: $(HARNESSES)

$(BUILD):
	mkdir -p $(BUILD)

# --- stencil exchange (vendored, unmodified) ------------------------------
$(BUILD)/noop_%.o: $(STENEX)/noop_%.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

# noop_modules.f aggregates the others, so they must compile first.
$(BUILD)/noop_modules.o: $(filter-out $(BUILD)/noop_modules.o,$(STENEX_OBJ))

# --- stubs ----------------------------------------------------------------
$(BUILD)/stubs.o: $(HARNESS)/stubs.f90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/stubs_driver.o: $(HARNESS)/stubs_driver.f90 $(BUILD)/stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

# --- vendored CMAQ science ------------------------------------------------
$(BUILD)/%.o: $(SRC)/%.F $(COMMON_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/zfdbc.o: $(SRC)/zfdbc.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/x_ppm.o: $(BUILD)/hppm.o $(BUILD)/xy_budget.o $(BUILD)/zfdbc.o
$(BUILD)/y_ppm.o: $(BUILD)/hppm.o $(BUILD)/xy_budget.o $(BUILD)/zfdbc.o
$(BUILD)/hadvppm.o: $(BUILD)/x_ppm.o $(BUILD)/y_ppm.o $(BUILD)/xy_budget.o

# --- harnesses ------------------------------------------------------------
$(BUILD)/harness_%.o: $(HARNESS)/harness_%.f90 $(COMMON_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_hppm: $(COMMON_OBJ) $(BUILD)/hppm.o $(BUILD)/harness_hppm.o
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/harness_vppm: $(COMMON_OBJ) $(BUILD)/vppm.o $(BUILD)/harness_vppm.o
	$(FC) $(FFLAGS) $^ -o $@

# PPM lives in vppm.F alongside VPPM, so this links the same object.
$(BUILD)/harness_ppm_coeffs: $(COMMON_OBJ) $(BUILD)/vppm.o $(BUILD)/harness_ppm_coeffs.o
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/harness_zfdbc: $(BUILD)/zfdbc.o $(BUILD)/harness_zfdbc.o
	$(FC) $(FFLAGS) $^ -o $@

# The whole horizontal-advection chain: driver, both sweeps, contravariant
# velocity, boundary condition and kernel -- all vendored, none of it stubbed.
# Only the boundary-concentration *reader* is replaced, since RDBCON is pure
# I/O and drags in the aerosol modules.
$(BUILD)/stub_rdbcon.o: $(HARNESS)/stub_rdbcon.f90 $(COMMON_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness_hadv: $(COMMON_OBJ) \
                       $(BUILD)/hppm.o $(BUILD)/zfdbc.o $(BUILD)/xy_budget.o \
                       $(BUILD)/hcontvel.o $(BUILD)/x_ppm.o $(BUILD)/y_ppm.o \
                       $(BUILD)/hadvppm.o $(BUILD)/stub_rdbcon.o \
                       $(BUILD)/harness_hadv.o
	$(FC) $(FFLAGS) $^ -o $@

# The vertical chain: driver plus kernel, both vendored.
$(BUILD)/zadvppmwrf.o: $(BUILD)/vppm.o

$(BUILD)/harness_zadv: $(COMMON_OBJ) $(BUILD)/vppm.o $(BUILD)/zadvppmwrf.o \
                       $(BUILD)/harness_zadv.o
	$(FC) $(FFLAGS) $^ -o $@

# --- horizontal diffusion -------------------------------------------------
# hcdiff3d.F calls DEFORM internally; both are vendored and unmodified.
$(BUILD)/hcdiff3d.o: $(BUILD)/deform.o

$(BUILD)/harness_hcdiff3d: $(COMMON_OBJ) $(BUILD)/deform.o $(BUILD)/hcdiff3d.o \
                           $(BUILD)/harness_hcdiff3d.o
	$(FC) $(FFLAGS) $^ -o $@

# The driver pulls in rho_j.F as well; all four are vendored and unmodified.
$(BUILD)/hdiff.o: $(BUILD)/hcdiff3d.o $(BUILD)/rho_j.o

$(BUILD)/harness_hdiff: $(COMMON_OBJ) $(BUILD)/deform.o $(BUILD)/hcdiff3d.o \
                        $(BUILD)/rho_j.o $(BUILD)/hdiff.o $(BUILD)/harness_hdiff.o
	$(FC) $(FFLAGS) $^ -o $@

# --- vertical diffusion: the two ACM2 solvers -----------------------------
# Pure linear algebra -- no meteorology, no I/O API, so the stub surface is just
# VGRD_DEFN and CGRID_SPCS, both of which the advection stubs already provide.
$(BUILD)/harness_tri: $(COMMON_OBJ) $(BUILD)/tri.o $(BUILD)/harness_tri.o
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/harness_matrix1: $(COMMON_OBJ) $(BUILD)/matrix1.o $(BUILD)/harness_matrix1.o
	$(FC) $(FFLAGS) $^ -o $@

# eddyx.F needs a Met_Data carrier. The real ASX_DATA_MOD is 1154 lines of
# land-surface and I/O bookkeeping; stubs_asx.f90 supplies only the twelve
# arrays and three parameters eddyx.F actually reads.
$(BUILD)/stubs_asx.o: $(HARNESS)/stubs_asx.f90 $(COMMON_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/eddyx.o: $(BUILD)/stubs_asx.o

$(BUILD)/harness_eddyx: $(COMMON_OBJ) $(BUILD)/stubs_asx.o $(BUILD)/eddyx.o \
                        $(BUILD)/harness_eddyx.o
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/harness_eddyx.o: $(BUILD)/stubs_asx.o

# The ACM2 driver. Its extra module surface -- species maps, deposition-velocity
# carrier, emission array -- is plain data; the modules that *compute* those are
# outside this port's scope.
$(BUILD)/stubs_vdiff.o: $(HARNESS)/stubs_vdiff.f90 $(COMMON_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/vdiffacmx.o: $(BUILD)/stubs_asx.o $(BUILD)/stubs_vdiff.o \
                      $(BUILD)/tri.o $(BUILD)/matrix1.o

$(BUILD)/harness_vdiff.o: $(BUILD)/stubs_asx.o $(BUILD)/stubs_vdiff.o

$(BUILD)/harness_vdiff: $(COMMON_OBJ) $(BUILD)/stubs_asx.o $(BUILD)/stubs_vdiff.o \
                        $(BUILD)/tri.o $(BUILD)/matrix1.o $(BUILD)/vdiffacmx.o \
                        $(BUILD)/harness_vdiff.o
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)
