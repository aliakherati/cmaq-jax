!> Golden harness for PPM -- the non-uniform-spacing reconstruction inside
!> vppm.F (vppm.F:396-544).
!>
!>   usage: harness_ppm_coeffs <input.bin> <output.bin>
!>
!> PPM is a separate external subroutine, not contained in VPPM, so it can be
!> called directly. That pins the reconstruction on its own, independent of the
!> flux-matching velocity adjustment that VPPM wraps around it -- which is what
!> lets the parabola be validated (chunk A0.6) before the adjustment exists
!> (A2.2).
!>
!> One call per process. PPM precomputes its mesh coefficients (ALPHA, CHI,
!> PSI, MU, NU, LAMBDA) from DS on the first call and SAVEs them
!> (vppm.F:450-468); a second call with a different NI or DS silently reuses
!> the first set.
!>
!> DT is in PPM's argument list but its body never reads it. Passed anyway so
!> the call matches the Fortran exactly.
!>
!>   input:  ni                (int32)
!>           dt                float32
!>           ds(ni)            float32
!>           cn(ni)            float32
!>   output: cr(ni), cl(ni), dc(ni), c6(ni)   float32
program harness_ppm_coeffs

   implicit none

   character(len=256) :: in_path, out_path
   integer :: unit_in, unit_out, ios
   integer :: ni
   real :: dt
   real, allocatable :: ds(:), cn(:), cr(:), cl(:), dc(:), c6(:)

   if (command_argument_count() /= 2) then
      write (*, '(a)') 'usage: harness_ppm_coeffs <input.bin> <output.bin>'
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

   read (unit_in) ni
   read (unit_in) dt

   allocate (ds(ni), cn(ni), cr(ni), cl(ni), dc(ni), c6(ni))

   read (unit_in) ds
   read (unit_in) cn
   close (unit_in)

   cr = 0.0
   cl = 0.0
   dc = 0.0
   c6 = 0.0

   call PPM(ni, dt, ds, cn, cr, cl, dc, c6)

   open (newunit=unit_out, file=trim(out_path), access='stream', &
         form='unformatted', status='replace', action='write')
   write (unit_out) cr
   write (unit_out) cl
   write (unit_out) dc
   write (unit_out) c6
   close (unit_out)

end program harness_ppm_coeffs
