!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): CH HM
!
subroutine a2y_kb_pp( ikibz, ncid )
  !
  ! Structure of ETSF library data: (fortran order, opp to netcdf order)
  !   kb_formfactors(  dims%max_number_of_coefficients,    &
  !   &                dims%number_of_kpoints,             &
  !   &                dims%max_number_of_projectors,      &
  !   &                dims%max_number_of_angular_momenta, &
  !   &                dims%number_of_atom_species )
  !   kb_formfactor_derivative( &
  !   &                dims%max_number_of_coefficients,    &
  !   &                dims%number_of_kpoints,             &
  !   &                dims%max_number_of_projectors,      &
  !   &                dims%max_number_of_angular_momenta, &
  !   &                dims%number_of_atom_species )
  !   kb_formfactor_sign( &
  !   &                dims%max_number_of_projectors,      &
  !   &                dims%max_number_of_angular_momenta, &
  !   &                dims%number_of_atom_species )
  ! Structure of YAMBO arrays, from PP_alloc:
  !   pp_kbs ( n_atom_species, pp_n_l_comp_max )
  !   pp_kb  ( ng_vec, n_atom_species, pp_n_l_comp_max )
  !   pp_kbd ( ng_vec, n_atom_species, pp_n_l_comp_max )
  !
  use netcdf
  use netcdf_data
  use pars,                 only : SP
  use D_lattice,            only : n_atomic_species
  use pseudo,               only : pp_n_l_times_proj_max,&
&                                  pp_kb,pp_kbd
  use wave_func,            only : wf_ncx
  !
  implicit none
  !
  integer,        intent(in)    :: ncid, ikibz
  ! 
  ! Work Space
  !
  integer               :: ig, ib, ip, ia
  real(SP),allocatable  :: pp_section(:,:,:)
  integer               :: start(5), count(5), varid, netcdf_error
  !
  allocate(pp_section( wf_ncx, pp_n_l_times_proj_max, n_atomic_species ))
  !
  ! Form factors 
  !
  start = [1,ikibz,1,1,1]
  count = [wf_ncx,1,1,pp_n_l_times_proj_max,n_atomic_species]
  !
  ! Form factors 
  !
  netcdf_error = nf90_inq_varid(ncid, "kb_formfactors", varid)
  netcdf_error = nf90_get_var(ncid, varid, pp_section)
  call netcdf_check(ncid, netcdf_error, varid, "kb_formfactors")
  forall( ig=1:wf_ncx, ia=1:n_atomic_species, ip=1:pp_n_l_times_proj_max ) &
&          pp_kb(ig,ia,ip,1) = pp_section(ig,ip,ia)

  !
  ! Form factor derivatives 
  !
  netcdf_error = nf90_inq_varid(ncid, "kb_formfactors_derivative", varid)
  netcdf_error = nf90_get_var(ncid, varid, pp_section)
  call netcdf_check(ncid, netcdf_error, varid, "kb_formfactors_derivative")
  forall( ig=1:wf_ncx, ia=1:n_atomic_species, ip=1:pp_n_l_times_proj_max ) &
&          pp_kbd(ig,ia,ip,1) = pp_section(ig,ip,ia)
  deallocate(pp_section)
  !
  return
end subroutine a2y_kb_pp



subroutine a2y_kb_pp_wfk( ikibz, ncid )
  !
  ! Structure of KB form factors in WFK file
  !   vkb(  max_number_of_coefficients,    
  !         lnmax,                         
  !         number_of_atom_species,        
  !         number_of_kpoints ) 
  !   vkbd( max_number_of_coefficients,    
  !         lnmax,                         
  !         number_of_atom_species,
  !         number_of_kpoints )
  !   vkbsign( lnmax,                      
  !            number_of_atom_species )
  !
  ! Structure of YAMBO arrays, from PP_alloc:
  !   pp_kbs ( n_atom_species, pp_n_l_times_proj_max )
  !   pp_kb  ( ng_vec, n_atom_species, pp_n_l_times_proj_max )
  !   pp_kbd ( ng_vec, n_atom_species, pp_n_l_times_proj_max )
  !
  use netcdf
  use netcdf_data
  use pars,                 only : SP
  use vec_operate,          only : sort
  use D_lattice,            only : n_atomic_species
  use pseudo,               only : pp_n_l_times_proj_max,&
&                                  pp_kb,pp_kbd
  use wave_func,            only : wf_ncx,wf_nc_k,wf_igk
  !
  implicit none
  !
  integer, intent(in)   :: ncid, ikibz
  ! 
  ! Work Space
  !
  integer               :: ig, ic, ib, ip, ia
  integer               :: wf_igk_indx(wf_nc_k(ikibz))
  integer               :: wf_igk_tmp(wf_nc_k(ikibz))
  integer               :: start(4), varid, netcdf_error
  real(SP),allocatable  :: pp_section(:,:,:)
  !
  allocate(pp_section( wf_ncx, pp_n_l_times_proj_max, n_atomic_species ))
  wf_igk_tmp = wf_igk(:wf_nc_k(ikibz),ikibz)
  call sort(wf_igk_tmp,indx=wf_igk_indx)
  start = [1,1,1,ikibz]
  !
  ! Form factors 
  !
  netcdf_error = nf90_inq_varid(ncid, "vkb", varid)
  netcdf_error = nf90_get_var(ncid, varid, pp_section, start)
  call netcdf_check(ncid, netcdf_error, varid, "vkb")
  !
  ! Sort the non-local components according to the g-vectors (same as the WF coefficients)
  do ia=1,n_atomic_species
    do ip=1,pp_n_l_times_proj_max
      do ic=1,wf_nc_k(ikibz)
        ig=wf_igk_indx(ic)
        pp_kb(ic,ia,ip,1) = pp_section(ig,ip,ia)
      end do
    end do
  end do
  !
  ! Form factor derivatives 
  !
  netcdf_error = nf90_inq_varid(ncid, "vkbd", varid)
  netcdf_error = nf90_get_var(ncid, varid, pp_section, start)
  call netcdf_check(ncid, netcdf_error, varid, "vkbd")
  !
  ! Sort the non-local components according to the g-vectors (same as the WF coefficients)
  do ia=1,n_atomic_species
    do ip=1,pp_n_l_times_proj_max
      do ic=1,wf_nc_k(ikibz)
        ig=wf_igk_indx(ic)
        pp_kbd(ic,ia,ip,1) = pp_section(ig,ip,ia)
      end do
    end do
  end do
  deallocate(pp_section)
  !
  return
end subroutine a2y_kb_pp_wfk
