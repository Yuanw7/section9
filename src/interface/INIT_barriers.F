!
! License-Identifier: GPL
!
! Copyright (C) 2014 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine INIT_barriers()
 !
 use drivers,    ONLY:l_acfdt,l_bse,l_bss,l_cohsex,l_elph_corr,l_gw0,l_phel_corr,&
&                     l_eval_collisions,l_life,l_ppa,l_mpa,l_real_time,l_sc_run,l_tddft,l_nl_optics,l_screen
 use pars,       ONLY:lchlen
 use D_lattice,  ONLY:i_time_rev,i_space_inv
 use R_lattice,  ONLY:Xk_grid_is_uniform
 use BS,         ONLY:BSK_mode
 use BS_solvers, ONLY:BSS_mode
 use electrons,  ONLY:n_spin,n_met_bands,n_full_bands
 use QP_m,       ONLY:QP_solver
 use it_tools,   ONLY:switch_off_runlevel
 implicit none
 ! 
 ! Additional logicals ...
 !
 logical :: metallic_system, bss_tddft,bse_bsfxc,&
            TR_or_SI,gw0_raxis,tddft_r_space,bss_diago,bse_sex,&
&           bss_haydock,gw0_ppa,gw0_mpa,gw0_cohsex,tddft_g_space,bss_inv,l_flag
 character(lchlen) :: always_on_runlevels,on_runlevels
 !
 ! ...
 !
 metallic_system=any(n_met_bands/=n_full_bands)
 !
 TR_or_SI=i_time_rev==1.or.i_space_inv==1
 !
 gw0_raxis=l_gw0.and..not.l_cohsex.and..not.l_ppa.and..not.l_mpa
 gw0_ppa=l_gw0.and.l_ppa
 gw0_mpa=l_gw0.and.l_mpa
 gw0_cohsex=l_gw0.and.l_cohsex
 !
 tddft_r_space=l_tddft.and.     l_bse
 tddft_g_space=l_tddft.and..not.l_bse
 !
 bse_sex  =trim(BSK_mode)=='SEX'
 bse_bsfxc=trim(BSK_mode)=='BSfxc'
 !
 bss_diago=l_bss.and.index(BSS_mode,'d')/=0
 bss_inv=l_bss.and.index(BSS_mode,'i')/=0
 bss_haydock=l_bss.and.index(BSS_mode,'h')/=0
 bss_tddft=l_bss.and.index(BSS_mode,'t')/=0
 !
 always_on_runlevels='setup rim_cut'
 !
 ! Note that switch_off_runlevel defines a barrier to all others runlevels
 ! except the ones specified. switch_off_runlevel switches off
 ! only already activated runlevels
 !
 ! RUNLEVELS
 !===========
 !
 if (l_real_time) then
   on_runlevels='HF_and_locXC cohsex em1s dipoles negf collisions el_el_corr el_ph_corr ph_el_corr '//&
&               'el_el_scatt el_ph_scatt el_photon_scatt el_photon_corr'
#if defined _PHEL
   on_runlevels=trim(on_runlevels)//' ph_el_scatt'
#endif
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_nl_optics) then
   on_runlevels='chi tddft bse ss ppa em1s dipoles em1d nloptics'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_eval_collisions) then
   on_runlevels='scrun HF_and_locXC cohsex em1s dipoles collisions el_el_corr el_ph_corr ph_el_corr el_el_scatt '//&
&               'el_ph_scatt el_photon_scatt el_photon_corr'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_sc_run) then
   on_runlevels='scrun HF_and_locXC cohsex em1s dipoles negf magnetic electric el_el_corr collisions '
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (gw0_cohsex) then
   call switch_off_runlevel('all',on_name='dyson gw0 em1s dipoles cohsex el_el_corr el_ph_corr ph_el_corr HF_and_locXC '//&
&                            trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_acfdt) then
   call switch_off_runlevel('all',on_name='acfdt HF_and_locXC '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_life) then
   call switch_off_runlevel('all',on_name='life dipoles em1d el_el_corr ph_el_corr el_photon_corr '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (gw0_raxis) then
   on_runlevels='dyson gw0 em1d el_el_corr el_ph_corr ph_el_corr el_photon_corr HF_and_locXC'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (gw0_ppa) then
   on_runlevels='dyson gw0 ppa em1d el_el_corr el_ph_corr ph_el_corr HF_and_locXC rim_w'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 if (gw0_mpa) then
   on_runlevels='dyson gw0 mpa em1d el_el_corr el_ph_corr HF_and_locXC'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 if (tddft_g_space) then
   call switch_off_runlevel('all',on_name='optics chi tddft '//trim(always_on_runlevels))
   goto 1
 endif
 if (tddft_r_space) then
   call switch_off_runlevel('all',on_name='optics bse bss tddft '//trim(always_on_runlevels))
   goto 1
 endif
 !
 l_flag=l_bss.or.l_bse
 if (l_flag) then
   on_runlevels='optics bse bss tddft em1s dipoles em1d ppa photolum rim_w'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
 if (l_screen) then
   on_runlevels='em1s dipoles em1d ppa X'
   call switch_off_runlevel('',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
   goto 1
 endif
 !
1 continue
 !
 ! ENVIROMENTS
 !=============
 !
 ! Non Uniform grids
 !===================
 if (.not.Xk_grid_is_uniform) then
   on_runlevels='optics chi bse bss negf'
   if (l_elph_corr) on_runlevels='optics chi bse bss el_ph_corr ph_el_corr dyson gw0'
   if (tddft_r_space.or.bse_sex.or.bse_bsfxc) BSK_mode='hartree'    
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
 endif 
 !
 ! SPIN
 !=============
 if (n_spin>2) then
   on_runlevels='optics chi bse bss em1s dipoles em1d ppa mpa HF_and_locXC cohsex dyson gw0 magnetic electric scrun'
   on_runlevels=trim(on_runlevels)//' negf collisions el_el_corr el_ph_corr ph_el_corr el_el_scatt el_ph_scatt'
   call switch_off_runlevel('all',on_name=trim(on_runlevels)//' '//trim(always_on_runlevels))
 endif 
 !
 ! MORE
 !======
 if ( (l_ppa.and.trim(QP_solver)=='s').or.l_cohsex ) QP_solver='n'
#if defined  _PHEL 
 if ( l_phel_corr.and.trim(QP_solver)=='s' ) QP_solver='n'
#endif
 !
end subroutine
