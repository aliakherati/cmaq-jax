!> Golden harness for HPPM -- CMAQ's 1-D uniform-spacing PPM kernel.
!>
!>   usage: harness_hppm <input.bin> <output.bin>
!>
!> Reads one case, makes exactly ONE call to the unmodified hppm.F, and writes
!> the updated concentrations plus the boundary fluxes.
!>
!> One call per process is deliberate. HPPM allocates its work arrays on the
!> first call, sized from NI/NJ/NSPCS, and SAVEs them (hppm.F:225-246). A second
!> call with a different shape silently reuses the first shape and returns
!> wrong numbers. Keeping it to one call per process makes that impossible.
!>
!> Binary format is little-endian stream, REAL(4) to match CMAQ's default REAL.
!> See scripts/generate_goldens.py for the writer.
!>
!>   input:  ni, nj, nspcs, ori_code            (4 x int32)
!>           dt, ds                             (2 x float32)
!>           con(1-SWP : ni+SWP, nspcs)         float32, Fortran order
!>           vel(ni+1)                          float32
!>   output: con(1-SWP : ni+SWP, nspcs)         float32, updated in place
!>           f_lo_in, f_lo_out, f_hi_in, f_hi_out   4 x float32(nspcs)
program harness_hppm

   use PA_DEFN, only: BUDGET_DIAG, BUDGET_HPPM

   implicit none

   ! Halo width. Must match the SWP parameter inside hppm.F:147.
   integer, parameter :: SWP = 3

   interface
      ! Copied verbatim from x_ppm.F:183-200. HPPM's arguments are
      ! assumed-shape, so an explicit interface is mandatory at the call site.
      subroutine HPPM(NI, NJ, CON, VEL, DT, DS, ORI, &
                      F_LO_IN, F_LO_OUT, F_HI_IN, F_HI_OUT)
         integer, parameter :: SWP = 3
         integer, intent(in) :: NI, NJ
         real, intent(inout) :: CON(1 - SWP:, 1:)
         real, intent(in) :: VEL(:)
         real, intent(in) :: DT
         real, intent(in) :: DS
         character, intent(in) :: ORI
         real, intent(out) :: F_LO_IN(:)
         real, intent(out) :: F_LO_OUT(:)
         real, intent(out) :: F_HI_IN(:)
         real, intent(out) :: F_HI_OUT(:)
      end subroutine HPPM
   end interface

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ni, nj, nspcs, ori_code
   real :: dt, ds
   character :: ori
   real, allocatable :: con(:, :), vel(:)
   real, allocatable :: f_lo_in(:), f_lo_out(:), f_hi_in(:), f_hi_out(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_hppm <input.bin> <output.bin>'
      stop 2
   end if
   call get_command_argument(1, in_path)
   call get_command_argument(2, out_path)

   open (newunit=unit_in, file=trim(in_path), access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
   if (ios /= 0) then
      write (*, '(a)') 'cannot open input: '//trim(in_path)
      stop 2
   end if

   read (unit_in) ni, nj, nspcs, ori_code
   read (unit_in) dt, ds

   ! 'C' sweeps a row (x-direction), 'R' sweeps a column (y-direction).
   ! HPPM only forwards ORI to SUBST_HI_LO_BND_PE, which our stub ignores,
   ! but we round-trip it so the golden records which sweep was run.
   if (ori_code == 0) then
      ori = 'C'
   else
      ori = 'R'
   end if

   allocate (con(1 - SWP:ni + SWP, nspcs), vel(ni + 1))
   allocate (f_lo_in(nspcs), f_lo_out(nspcs), f_hi_in(nspcs), f_hi_out(nspcs))

   read (unit_in) con
   read (unit_in) vel
   close (unit_in)

   ! HPPM leaves the boundary-flux outputs untouched unless BUDGET_HPPM is set.
   BUDGET_DIAG = .true.
   BUDGET_HPPM = .true.
   f_lo_in = 0.0
   f_lo_out = 0.0
   f_hi_in = 0.0
   f_hi_out = 0.0

   call HPPM(ni, nj, con, vel, dt, ds, ori, &
             f_lo_in, f_lo_out, f_hi_in, f_hi_out)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) con
   write (unit_out) f_lo_in
   write (unit_out) f_lo_out
   write (unit_out) f_hi_in
   write (unit_out) f_hi_out
   close (unit_out)

end program harness_hppm
