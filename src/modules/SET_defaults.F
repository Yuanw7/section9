!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine SET_defaults(INSTR,IND,OD,COM_DIR)
 !
 use debug,      ONLY:dbg
 use pars,       ONLY:SP,cZERO
 use C_driver,   ONLY:code_branch
 use units,      ONLY:HA2EV,FS2AUT,kWCMm22AU
 use LOGO,       ONLY:ID_logo,ID_logo_stderr
 use global_XC,  ONLY:EXT_NONE
 use X_m,        ONLY:current_iq,self_detect_E_range,X_FILL_UP_matrix_only,eps_2_alpha,l_drude,&
&                     alpha_dim,use_X_DbGd,X_DbGd_nkpts,Chi_mode,skip_cond_sum_rule,&
&                     q_plus_G_direction,Q_Shift_Order,&
&                     l_X_terminator,X_terminator_E,global_gauge,&
&                     Chi_linalg_mode,X_use_lin_sys,X_use_gpu,X_DbGd_percentual
 use DIPOLES,    ONLY:use_covariant_approach,use_real_space_approach,&
&                     Vnl_commutator_warning,use_shifted_grids_approach,use_g_space_approach,&
&                     compute_P2_dipoles,compute_Spin_dipoles,compute_Orb_dipoles
 use QP_m,       ONLY:QP_dSc_steps,QP_n_W_freqs,QP_G_Zoom_treshold,&
&                     QP_dSc_test,QP_solver,QP_G_damp,QP_dSc_delta,&
&                     QP_cg_percent,QP_n_states,SC_E_threshold, &
&                     QP_Sc_steps,QP_G_er,QP_G_dr,SC_band_mixing,&
&                     COHSEX_use_empties,On_Mass_Shell_approx,&
&                     Vnlxc_kind,Vxc_kind,l_extended_output,l_GW_terminator,&
&                     GW_terminator_E
 use QP_CTL_m,   ONLY:QP_ctl_user,QP_ctl_DB_user
 use functions,  ONLY:bose_E_cut
 use D_lattice,  ONLY:i_space_inv,inv_index,n_atoms_species_max,n_atomic_species,mag_syms,&
&                     input_Tel_is_negative,non_periodic_directions,lattice,Bose_Temp,    &
&                     molecule_position
 use pseudo,     ONLY:pp_n_l_times_proj_max,pp_kbv_dim
 use R_lattice,  ONLY:n_g_shells,ng_closed,bse_scattering,coll_scattering,&
&                     Xk_grid_is_uniform,RIM_id_epsm1_reference,RIM_epsm1,&
&                     RIM_anisotropy,RIM_ng,RIM_W_ng,rimw_type,RIM_n_rand_pts,nqibz,q0_def_norm,&
&                     cutoff_presets,k_map,qindx_S_max_Go
 use electrons,  ONLY:n_spin,n_sp_pol,n_spinor,filled_tresh,l_spin_orbit,n_spin_den,eval_magn
 use parallel_m, ONLY:ncpu,CPU_str_reset,CREATE_hosts_COMM
 use com,        ONLY:isec,depth,secnm,previous_secmode,of_name,n_ofs_defined,of_opened,of_unit,max_open_ofs,&
&                     more_io_path,core_io_path,com_path,repfile,grid_paths,fat_log,exp_user
 use stderr,     ONLY:win_size,tty_size,logfile,set_real_printed_length,log_as_a_file
 use LIVE_t,     ONLY:log_line_to_dump,log_line,nhash,ct
 use wave_func,  ONLY:wf_ng,wf_norm_test,wf_nb_io,wf_nb_io_groups,WF,WF_buffer,WF_buffered_IO
 use FFT_m,      ONLY:fft_dim_loaded,fft_size,fft_dim,fft_multiplier
 use IO_m,       ONLY:io_reset,max_io_units,serial_number,frag_WF
 use BS_solvers, ONLY:BSS_mode,BSS_n_freqs,BSS_er,BSS_dr,&
&                     BSS_q0,Haydock_threshold,Haydock_iterIO,Haydock_iterMAX,BSS_uses_DbGd,BSS_Wd,&
&                     BSS_damp_reference,BSS_Vnl_included,BSS_uses_GreenF,BSS_inversion_mode,&
&                     BSS_perturbative_width,K_INV_EPS,K_INV_PL,K_INV_PI_PH
 use descriptors,ONLY:IO_desc_reset
#if defined _SLEPC && !defined _NL
 use BS_solvers, ONLY:BSS_slepc_extraction,BSS_slepc_ncv,BSS_slepc_tol,BSS_slepc_target_E,BSS_slepc_maxit,&
 &                    BSS_slepc_precondition,BSS_slepc_approach
#endif
 use BS,         ONLY:BS_n_g_W,BS_eh_en,BS_identifier,BS_q,BS_eh_win,MAX_BSK_LIN_size,&
&                     BS_K_dim,BS_not_const_eh_f,BSK_mode,l_BSE_kernel_complete,&
&                     BS_K_is_ALDA,BSE_mode,BSE_prop,BSE_L_kind,BS_K_cutoff,BS_perturbative_SOC,BSK_IO_mode
 use TDDFT,      ONLY:FXC_type,FXC_n_g_corr,ALDA_cut_scheme,&
&                     FXC_per_memstps,FXC_LRC_alpha,FXC_PF_alpha,FXC_LRC_beta,FXC_SVD_digits,&
&                     FXC_is_retarded,TDDFT_mode
 use ACFDT,      ONLY:ACFDT_n_lambda,ACFDT_n_freqs,ACFDT_E_range
#if defined _ELPH 
 use ELPH,       ONLY:ph_modes,elph_nb,gsqF_energy_steps,eval_G_using_KK,use_PH_DbGd,EkplusQ_mode,gkkp_db,&
&                     elph_Ham_bands,elph_Ham_ik,elph_branches,RES_tresh,FAN_deltaE_treshold,DW_deltaE_treshold
#endif
 use zeros,      ONLY:zero_norm,k_iku_zero,k_rlu_zero,G_iku_zero,G_mod_zero,zero_dfl
 use memory,     ONLY:MEMs,LARGE_MEMs,N_MEM_max,N_MEM_SAVE_max,MEM_element_init
 use xc_functionals,  ONLY:GS_xc_FUNCTIONAL,GS_xc_KIND,GS_exx_FRACTION,GS_exx_SCREENING 
 use BS,         ONLY:l_BS_anomalous_Hall
 use PHOTOLUM,   ONLY:PL_weights
#if defined _SC 
 use SC,         ONLY:SC_iterations,SC_rho_threshold,SC_potential,OEP_approx,l_NSC_shot,&
&                     compatible_SC_DB,SC_cycle_mixing,SC_fft_size,found_SC_DB,&
&                     l_Mean_Potential,SC_up_W_iters,l_SC_diagonal,&
&                     l_SC_nl_mix,SC_kind,SC_perturbation,SC_desc,SC_neq_kind,SC_last_coh_it
#endif
#if defined _SC || defined _RT || defined _QED || defined _NL
 use collision_ext,  ONLY:COLLISIONS_naming,COLLISIONS_cutoff,COLLISIONS_CV_only
 use hamiltonian,    ONLY:H_potential
#endif
#if defined _RT || defined _YPP_RT
 use RT_output_m,ONLY:n_RT_headings,N_RT_o_files,RT_o_file_long,RT_o_file_short,RT_o_file_is_open,&
&                     RT_headings,YPP_append,Gr_desc,TIME_STEP_desc,RT_desc
 use RT_control, ONLY:CARR_RT_IO_t,OBS_RT_IO_t,OUTPUT_RT_IO_t,Gless_RESTART_RT_IO_t,SAVE_G_history,RT_ctl_db,&
&                     CACHE_OBS_last_point,CACHE_OBS_INTERVAL_time,CACHE_OBS_INTERVAL_time_INPUT,   &
&                     CACHE_OBS_steps,CACHE_OBS_steps_now,RT_NAN_found
 use RT_lifetimes,ONLY:RT_life_extrap_times
 use real_time,  ONLY:RT_step,NE_steps,NE_time,Integrator_name,RT_deph_range,&
&                     Gr_kind,two_alpha,RAD_LifeTime,Phase_LifeTime,&
&                     NE_tot_time,NE_MEM_treshold,RT_eh_en,l_update_SOC,&
&                     DbGd_EE_percent,RT_scatt_tresh,integrator_step,RT_nk,&
&                     i_MEM_old,i_MEM_prev,i_MEM_now,G_MEM_steps,RT_step_manual_prefactor,&
&                     RT_deph_deg_thresh,RT_deph_cv_only,NE_time_step_update_manual,&
&                     NE_time_step_update_last_point,NE_time_step_update_jump,NE_time_step_update_jump_INPUT, &
&                     NE_initial_time_step_update,NE_step_update_treshold
 use fields,     ONLY:Efield,Efield_reset,n_ext_fields_max,n_ext_fields,A_vecpot_reset,   &
&                     A_tot,A_ind,A_ext
 use plasma,     ONLY:PLASMA_redux_percent
#endif
#if defined _NL
 use nl_optics,  ONLY:NL_correlation,NL_er,n_frequencies,NL_damping,NL_LRC_alpha,         &
&                     NL_verb_name,NL_bands,n_order,NL_desc
 use fields,     ONLY:Efield,Efield_reset
#endif
#if defined _SC
 use magnetic,    ONLY:MAG_B,MAG_radius,MAG_hamiltonian_type,MAG_landau,MAG_pauli,&
&                      MAG_gauge,MAG_psi,MAG_theta,phase_trick
 use electric,    ONLY:l_P_periodic,l_force_SndOrd
#endif
#if defined _YAML_OUTPUT
 use com,         ONLY:depth_yaml
#endif
 use timing_m,    ONLY:timing_allocate,nclockx
 use openmp,      ONLY:OPENMP_initialize,OPENMP_update,master_thread,omp_is_off
 use cuda_m,      ONLY:cuda_setup,have_cuda
 !
 implicit none
 !
 character(*) :: INSTR,IND,OD,COM_DIR
 !
 ! Work Space 
 !
 integer           :: i1,i2
 !
 ! Printed reals format lengths 
 !
 include 'branch.inc'
 !
 call set_real_printed_length()
 !
 ! Fat Log?
 !
 fat_log    = .FALSE.
 if (index(INSTR,'fatlog')>0) fat_log=.TRUE.
 !
 ! Experienced User
 !
 exp_user    = .FALSE.
 if (index(INSTR,'expuser')>0) exp_user=.TRUE.
 !
 ! CPU structures
 !
 call CPU_str_reset()
 !
 ! OpenMP
 !
#if defined _OPENMP
 omp_is_off    = .FALSE.
 if (index(INSTR,'noopenmp')>0) omp_is_off=.TRUE.
#else
 omp_is_off    = .TRUE.
#endif
 !
 call OPENMP_initialize( )
 call OPENMP_update(master_thread)
 !
 ! Node name
 !
 call CREATE_hosts_COMM( )
 !
 ! CUDA environment defs
 !
 call cuda_setup()
 !
 ! Stack Size 
 !
 call remove_stack_limit()
 !
 ! Debug
 !
 call dbg('reset')
 !
 ! Clocks
 !
 call timing_allocate(nclockx)
 !
 call MEM_element_init(MEMs,N_MEM_max)
 call MEM_element_init(LARGE_MEMs,N_MEM_SAVE_max)
 !
 ! ZEROs 
 !
 zero_norm =zero_dfl
 k_iku_zero=zero_dfl
 k_rlu_zero=zero_dfl
 G_iku_zero=zero_dfl
 G_mod_zero=1.E-5
 !
 ! PATHS  ...
 !
 core_io_path=IND
 more_io_path=OD
 com_path=COM_DIR
 !
 ! WFs fragmentation, this is for the interfaces
 !
 frag_WF = (.not.index(INSTR,'nodbfr')>0 .or. index(INSTR,'fragnb')>0)
 !
 ! Logical Setup (Mainly for interfaces and ypp. As far as yambo is concerned this call is done in init.F)
 !
 call SET_logicals()
 !
 ! TTY size 
 !
 call win_size(tty_size)
 call ct(INIT=.TRUE.)
 log_as_a_file=ncpu>1.or.tty_size<0
 !
 !I/O 
 !
 serial_number=0
 !
 !com
 !
 isec=0
 depth=-1
 secnm=' '
 previous_secmode=' '
 of_name=' '
 n_ofs_defined=0
 of_opened=' '
 of_unit=0
 of_unit(max_open_ofs)=-1
#if defined _YAML_OUTPUT
 depth_yaml=0
#endif
 !
 !LOGO
 !
 ID_logo=-1
 ID_logo_stderr=-1
 !
 !LOG/REP files
 !
 repfile=" "
 logfile=" "
 !
 !Timing
 !
 log_line_to_dump=.FALSE.
 log_line=' '
 nhash=40
 !
 !functions
 !
 bose_E_cut=0.1_SP
 !
 !D_lattice
 !
 input_Tel_is_negative=.FALSE.
 non_periodic_directions='none'
 molecule_position=0._SP
 lattice='Unknown'
 Bose_Temp=-1./HA2EV
 !
 ! R_lattice
 !
 n_g_shells=0
 nqibz=0
 ng_closed=0
 coll_scattering=.FALSE.
 bse_scattering=.FALSE.
 Xk_grid_is_uniform=.TRUE.
 q0_def_norm=1.E-5_SP
 k_map%g0_idx  =-1
 k_map%q_step  =-1
 k_map%max_kdir= 0
 qindx_S_max_Go=-1
 !
 ! RIM
 !
 RIM_id_epsm1_reference=0
 RIM_epsm1=0.
 RIM_anisotropy=0._SP
 RIM_ng=0
 RIM_n_rand_pts=0
 !
 ! RIM-W
 !
 RIM_W_ng=0
 RIMW_type="default"
 !
 ! CUTOFF
 !
 call cutoff_presets()
 !
 ! D_lattice 
 !
 n_atoms_species_max=0
 n_atomic_species=0
 i_space_inv=-1
 inv_index=0
 mag_syms=.FALSE.
 !
 ! Pseudo
 !
 pp_n_l_times_proj_max=0
 pp_kbv_dim=0
 !
 ! Electrons
 !
 n_spin=1
 n_sp_pol=1
 n_spinor=1
 n_spin_den=1
 l_spin_orbit       = .FALSE.
 filled_tresh       =0.00001_SP
 !
 ! Magnetization and density
 !
 eval_magn      = .FALSE.
 !
 !wave_func
 !
 WF%b=0
 WF%k=0
 WF%space=' '
 WF_buffer%b=0
 WF_buffer%k=0
 WF_buffer%space=' '
 WF_buffered_IO=.FALSE.
 wf_ng=0
 wf_norm_test=.TRUE.
 wf_nb_io=0
 wf_nb_io_groups=1
 !
 !FFT
 !
 fft_dim_loaded=0
 fft_size=0
 fft_dim=0
 fft_multiplier=(/1,1,1/)
 !
 do i1=1,max_io_units
   call io_reset(i1)
 enddo
 !
 !X
 !
 Chi_mode=' '
 Chi_linalg_mode="lin_sys"
 X_use_lin_sys=.FALSE.
 X_use_gpu=have_cuda
 current_iq=0
 X_DbGd_nkpts=0
 self_detect_E_range=.FALSE.
 X_FILL_UP_matrix_only=.FALSE.
 use_shifted_grids_approach = .FALSE.
 use_covariant_approach     = .FALSE.
 use_real_space_approach    = .FALSE.
 use_g_space_approach       = .TRUE.
 compute_P2_dipoles         = .FALSE.
 compute_Spin_dipoles       = .FALSE.
 compute_Orb_dipoles        = .FALSE.
 use_X_DbGd=.FALSE.
 X_DbGd_percentual=-1.
 eps_2_alpha=1._SP
 alpha_dim='adim'
 global_gauge='length'
 grid_paths=' '
 skip_cond_sum_rule=.FALSE.
 Vnl_commutator_warning=.FALSE.
 q_plus_G_direction=0._SP
 Q_Shift_Order=1
 l_X_terminator=.FALSE.
 X_terminator_E=40._SP/HA2EV
 !
 !QPm
 !
 QP_n_states=0
 QP_dSc_steps=2
 QP_G_Zoom_treshold=0._SP
 QP_Sc_steps=100
 QP_n_W_freqs=100
 QP_dSc_test=.FALSE.
 QP_solver=' '
 QP_G_damp=0.1/HA2EV
 QP_dSc_delta=0.1/HA2EV
 QP_G_er=(/-10._SP/HA2EV,10._SP/HA2EV/)
 QP_G_dr=0.1/HA2EV
 QP_cg_percent=100._SP
 COHSEX_use_empties=.FALSE.
 On_Mass_Shell_approx=.FALSE.
 SC_E_threshold=0.01/HA2EV
 SC_band_mixing=100._SP
 Vnlxc_kind='Fock'
 Vxc_kind='DFT'
 l_extended_output=.FALSE.
 l_GW_terminator=.FALSE.
 GW_terminator_E=1.5_SP
 !
 ! QP_ctl control
 !
 do i1=1,3
   QP_ctl_DB_user(i1)%INTERP_shell_factor=20._SP
   QP_ctl_DB_user(i1)%INTERP_N_neigh=1
   QP_ctl_DB_user(i1)%INTERP_DbGd_mode="NN"
   QP_ctl_DB_user(i1)%action="none"
   do i2=1,3
     QP_ctl_user(i1,i2)%E=(/0.,1.,1./)
     QP_ctl_user(i1,i2)%Wc=0._SP
     QP_ctl_user(i1,i2)%Wv=0._SP
     QP_ctl_user(i1,i2)%Wc_E_ref=0._SP
     QP_ctl_user(i1,i2)%Wv_E_ref=0._SP
     QP_ctl_user(i1,i2)%Wc_dos=0._SP
     QP_ctl_user(i1,i2)%Wv_dos=0._SP
     QP_ctl_user(i1,i2)%Z=(1._SP,0._SP)
   enddo
 enddo
 !
#if defined _RT
 !
 ! RT control
 !
 RT_ctl_db="none"
 !
#endif
 !
 ! BS/BSS
 !
 BS_n_g_W=-1
 BS_eh_en=(/-1._SP,-1._SP/)/HA2EV
 BS_identifier=0
 BS_q=1
 BS_eh_win=100._SP
 BSE_L_kind='BAR'
 BSE_mode='resonant'
 BSE_prop='abs'
 BSK_mode=' '
 BSK_IO_mode='2D_standard'
 l_BSE_kernel_complete=.false.
 TDDFT_mode=' '
 ALDA_cut_scheme="none"
 BSS_Wd=cZERO
 l_drude=.false.
 BS_K_dim=0
 BS_K_cutoff=0.0000_SP
 BSS_mode=' '
 BSS_inversion_mode='pf'
 BSS_er=(/0._SP,10._SP/)/HA2EV
 BSS_dr=0.1_SP/HA2EV
 BSS_q0=(/1._SP,0._SP,0._SP/)
 BSS_uses_DbGd=.FALSE.
 BSS_damp_reference=0._SP
 BS_K_is_ALDA=.FALSE.
 BS_not_const_eh_f=.FALSE.
 BS_perturbative_SOC=.FALSE.
 Haydock_threshold=-0.02_SP
 Haydock_iterMAX=1000
 Haydock_iterIO=100
 BSS_Vnl_included=.FALSE.
 BSS_uses_GreenF=.FALSE.
 BSS_perturbative_width=.FALSE.
 K_INV_PI_PH%what = "PH_PI"
 K_INV_PL%what     = "PLS"
 K_INV_PL%treshold =0.5_SP
 K_INV_EPS%what    = "EPS"
 K_INV_EPS%treshold=0.01_SP
 K_INV_EPS%max_treshold=40._SP
 BSS_n_freqs=100
 MAX_BSK_LIN_size=45000*45000 ! 46341 is the sqare root of the maximum integer in IP: 2147483647
 !
#if defined _SLEPC && !defined _NL
 BSS_slepc_approach='none'
 BSS_slepc_precondition='none'
 BSS_slepc_extraction='ritz'
 BSS_slepc_maxit=0
 BSS_slepc_ncv=0
 BSS_slepc_tol=1E-6_SP
 BSS_slepc_target_E=0._SP
#endif
 !
 ! TDDFT
 !
 FXC_type='rpa'
 FXC_n_g_corr=1
 FXC_per_memstps=100._SP
 FXC_LRC_alpha=0._SP
 FXC_PF_alpha="CUR"
 FXC_LRC_beta=0._SP
 FXC_SVD_digits=0
 FXC_is_retarded=.FALSE.
 !
 ! ACFDT
 !
 ACFDT_n_lambda=1
 ACFDT_n_freqs=10
 ACFDT_E_range=(/100.,1000./)/HA2EV
 !
 ! xc_functionals  
 !
 GS_xc_FUNCTIONAL=-1             ! unknow 
 GS_xc_KIND=-1                   ! unknow 
 GS_exx_FRACTION=0.0             ! no EXX part  
 GS_exx_SCREENING=0.0            ! no screening
 !
 l_BS_anomalous_Hall=.false.
 PL_weights=(/1._SP,1._SP,1._SP/)
 !
#if defined _ELPH 
 !
 ! ELPH
 !
 ph_modes=0
 elph_nb=0
 elph_branches=0
 gsqF_energy_steps=2
 eval_G_using_KK=.FALSE.
 elph_Ham_bands=0
 elph_Ham_ik=1
 RES_tresh=0.01_SP
 DW_deltaE_treshold=1.E-6_SP/HA2EV   ! This is a cutoff on electronic energy differences
 FAN_deltaE_treshold=1.E-6_SP/HA2EV  ! This is a cutoff on phonon energies
 use_PH_DbGd=.FALSE.
 EkplusQ_mode='interp' ! E(k+q) energies (interp | dftp )
 gkkp_db='gkkp'        ! gkkp_db ( gkkp | gkkp_expanded | genFroh )
 !
#endif
 !
#if defined _SC || defined _RT || defined _QED || defined _NL
 !
 call COLLISIONS_naming( 0 , "all" )
 !
 ! COLLISIONS I/O
 !
# if defined _PAR_IO
 COLLISIONS_cutoff=0.0000_SP
# else
 COLLISIONS_cutoff=0.0005_SP
# endif
 COLLISIONS_CV_only=.FALSE.
 !
 H_potential="NONE"
 !
#endif
 !
#if defined _SC
 ! SC
 !
 SC_fft_size=0
 SC_iterations=100
 SC_cycle_mixing=0.5_SP
 SC_rho_threshold=1.E-5_SP
 SC_up_W_iters=0
 SC_potential='NONE'
 SC_kind=0
 SC_perturbation=EXT_NONE
 OEP_approx='n'
 compatible_SC_DB=.FALSE.
 found_SC_DB=.FALSE.
 l_Mean_Potential=.FALSE.
 l_SC_diagonal=.FALSE.
 l_SC_nl_mix =.FALSE.
 l_NSC_shot  =.FALSE.
 !
 SC_neq_kind   ='none'
 SC_last_coh_it=50
 !
 call IO_desc_reset(SC_desc)
 !
 l_P_periodic   =.FALSE.
 l_force_SndOrd =.FALSE.
 !
#endif
 !
#if defined _RT
 !
 ! Real Time
 !
 n_ext_fields=0
 do i1=1,n_ext_fields_max
   call Efield_reset(Efield(i1))
   Efield(i1)%t_initial=RT_step
 enddo
 !
 call A_vecpot_reset(A_tot)
 call A_vecpot_reset(A_ext)
 call A_vecpot_reset(A_ind)
 !
 ! RT dynamics 
 !
 l_update_SOC=.FALSE.
 !
 ! RT current 
 !
 RT_step_manual_prefactor=0._SP
 RT_step=0.010*FS2AUT
 NE_steps=1
 NE_time=0._SP
 NE_tot_time= 1000._SP*FS2AUT
 !
 NE_time_step_update_last_point = 1
 NE_time_step_update_jump       = 0._SP
 NE_time_step_update_jump_INPUT  = 0._SP
 NE_initial_time_step_update    = 0._SP
 NE_step_update_treshold        = 0.1_SP
 NE_time_step_update_manual     =-1.0_SP
 !
 CACHE_OBS_steps      = 0
 CACHE_OBS_steps_now  = 0
 CACHE_OBS_last_point = 0
 CACHE_OBS_INTERVAL_time       = 0._SP
 CACHE_OBS_INTERVAL_time_INPUT  = 0._SP
 !
 G_MEM_steps = 2
 i_MEM_old   = 0
 i_MEM_prev  = 0
 i_MEM_now   = 1
 !
 integrator_step=1
 Integrator_name='RK2'
 !
 RAD_LifeTime = 0._SP
#if defined _NL
 RAD_LifeTime = -1._SP*FS2AUT  ! Automatic set equal to the dephasing
#endif
 Phase_LifeTime = 0._SP
 RT_deph_deg_thresh = 1.E-5_SP
 RT_deph_cv_only =.FALSE.
 RT_deph_range=-1._SP
 !
 NE_MEM_treshold=0._SP
 RT_nk=0
 RT_eh_en=0._SP
 RT_scatt_tresh=-1._SP/HA2EV
 PLASMA_redux_percent=100._SP
 DbGd_EE_percent=0._SP
 !
 ! Scattering
 !
 RT_life_extrap_times(1)=-1._SP*FS2AUT
 RT_life_extrap_times(2)=-1._SP*FS2AUT
 !
 ! G_retarded
 !
 Gr_kind="HS"
 two_alpha=1
 !
 ! RT_ctl
 !
 OBS_RT_IO_t%INTERVAL_time_INPUT              =2._SP*FS2AUT
 CARR_RT_IO_t%INTERVAL_time_INPUT             =2._SP*FS2AUT
 OUTPUT_RT_IO_t%INTERVAL_time_INPUT           =5._SP*FS2AUT
 Gless_RESTART_RT_IO_t%INTERVAL_time_INPUT    =1._SP*FS2AUT
 SAVE_G_history=.FALSE.
 RT_NAN_found  =.FALSE.
 !
 ! RT_output
 !
 n_RT_headings=0
 N_RT_o_files=0
 RT_o_file_is_open=.FALSE.
 RT_o_file_long= " "
 RT_o_file_short= " "
 RT_headings= " "
 YPP_append=" "
 !
 call IO_desc_reset(RT_desc)
 call IO_desc_reset(Gr_desc)
 call IO_desc_reset(TIME_STEP_desc)
 !
#endif
 !
#if defined _NL
 !
 NE_tot_time   =  -1._SP*FS2AUT 
 Integrator_name='INVINT'
 NL_bands      =(/0,0/)
 NL_correlation="IPA"
 NL_er         =(/-1._SP,-1._SP/)/HA2EV
 n_frequencies =1
 n_order    = 0
 NL_damping    =0.2_SP/HA2EV
 NL_LRC_alpha  =0._SP
 !
 do i1=1,n_ext_fields_max
   call Efield_reset(Efield(i1))
   Efield(i1)%t_initial=RT_step
   Efield(i1)%frequency=0.1_SP/HA2EV
   Efield(i1)%intensity=0.0_SP
   Efield(i1)%FWHM=2._SP*FS2AUT
   Efield(i1)%ef_name="none"
   Efield(i1)%versor =(/0._SP,0._SP,0._SP/)
   Efield(i1)%ef_pol ="linear"
 enddo
 !
 Efield(1)%versor =(/0._SP,0._SP,0._SP/)
 Efield(1)%intensity=1000._SP*kWCMm22AU
 Efield(1)%ef_name="SOFTSIN"
 NL_verb_name='high'
 !
 call IO_desc_reset(NL_desc)
 !
#endif
 !
#if defined _SC
 !
 ! Magnetic
 !
 MAG_B=0._SP
 MAG_radius=0._SP
 MAG_hamiltonian_type=' '
 MAG_pauli=.FALSE.
 MAG_landau=.FALSE.
 MAG_gauge='SYMM'
 MAG_psi=0._SP
 MAG_theta=0._SP
 phase_trick=.FALSE.
 !
#endif
 !
end subroutine
