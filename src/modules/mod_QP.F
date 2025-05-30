!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
module QP_m
 !
 use descriptors,    ONLY:IO_desc,IO_desc_duplicate,IO_desc_reset
 use pars,           ONLY:SP,schlen,cZERO
 !
#include<memory.h>
 !
 ! Mixing for non perturbative calculations
 integer           :: SC_bands_mixed
 real(SP)          :: SC_band_mixing
 real(SP)          :: SC_E_threshold
 !
 ! Sigma_c
 integer           :: QP_dSc_steps
 integer           :: QP_Sc_steps
 integer           :: QP_n_W_freqs
 integer           :: QP_n_W_freqs_redux
 integer           :: QP_nb
 integer           :: QP_nk
 integer           :: QP_n_states
 integer           :: QP_ng_Sc
 integer           :: QP_ng_Sx
 integer           :: QP_ng_SH
 integer           :: QP_ng_Vxc
 integer           :: QP_n_G_bands(2)
 logical           :: COHSEX_use_empties
 logical           :: l_QP_symmetrize=.FALSE.
 logical           :: l_GW_terminator=.FALSE.
 logical           :: l_QP_Expand
 logical           :: use_GreenF_to_eval_QP
 logical           :: use_GreenF_Zoom
 logical           :: GF_is_causal
 character(schlen) :: QP_solver
 character(schlen) :: GF_energy_range_mode="centered"
 real(SP)          :: QP_G_Zoom_treshold
 real(SP)          :: QP_time_order_sign=-1 ! T-ordered. Set to +1 (causal) in QP_SET_temperature_pre_factor
 real(SP)          :: QP_G_damp
 real(SP)          :: QP_G_er(2)
 real(SP)          :: QP_G_dr(2)
 real(SP)          :: QP_cg_percent
 character(schlen) :: GW_terminator_Kind="none"
 real(SP)          :: GW_terminator_E
 real(SP)   ,allocatable:: QP_G_amplitude_integral(:)
 real(SP)   ,allocatable:: QP_G_info(:,:,:)
 real(SP)   ,allocatable:: QP_G_zoom_er(:,:)
 real(SP)   ,allocatable:: QP_W_er(:,:)
 real(SP)   ,allocatable:: QP_W_dr(:,:)
 complex(SP),allocatable:: QP_dSc(:,:)
 complex(SP),allocatable:: QP_W(:,:,:,:)
 complex(SP),allocatable:: QP_Vnl_xc(:)
 complex(SP),allocatable:: QP_Vnl_sex(:)
 complex(SP),allocatable:: QP_Sc(:,:)
 complex(SP),allocatable:: QP_Vxc(:)
 integer,    allocatable:: QP_solver_state(:)
 integer,    allocatable:: QP_table(:,:)
 integer,    allocatable:: QP_W_table(:,:)
 logical,    allocatable:: QP_state(:,:)
 !
 ! Solver...
 real(SP)          :: QP_dSc_delta
 logical           :: QP_dSc_test
 logical           :: On_Mass_Shell_approx
 ! 
 ! QED
 !
 integer           :: QP_QED_ng = 1
 !
 ! Logicals used for reporting
 !
 logical           :: report_Vxc
 logical           :: report_Vnlxc
 logical           :: report_Sc
 logical           :: report_dSc
 logical           :: l_extended_output
 logical           :: l_QP_eval_GF=.FALSE.
 logical           :: l_QP_eval_Z=.FALSE.
 character(schlen) :: Vnlxc_kind
 character(schlen) :: Vxc_kind
 !
 type QP_t
   !
   integer,    allocatable:: table(:,:)
   real(SP)   ,allocatable:: k(:,:)
   real(SP)   ,allocatable:: wk(:)
   real(SP)   ,allocatable:: E_bare(:)
   complex(SP),allocatable:: E(:)
   complex(SP),allocatable:: Z(:)
   integer  :: nk
   integer  :: nb
   integer  :: n_states
   !
   type(IO_desc) :: desc
   !
   ! Green Function and Self-Energy
   !
   integer                :: GreenF_n_steps
   complex(SP),allocatable:: S_total(:,:)
   complex(SP),allocatable:: GreenF(:,:)
   complex(SP),allocatable:: GreenF_W(:,:)
   !
 end type QP_t
 !
 ! Interface
 !
 interface
   !
   subroutine XCo_Hartree_Fock(E,k,Xk,q,mode)
     use pars,          ONLY:SP,schlen,DP,pi
     use electrons,     ONLY:levels,spin_occ,spin
     use R_lattice,     ONLY:qindx_S,bz_samp
     implicit none
     type(levels) ::E       
     type(bz_samp)::k,Xk,q
     character(*),optional ::mode
   end subroutine XCo_Hartree_Fock
   !
   subroutine QP_states_simmetrize(en,what,V_real,V_complex,state_is_2do,warn_me)
     use pars,          ONLY:SP
     use electrons,     ONLY:levels
     type(levels)          ::en
     character(*),optional ::what
     real(SP),    optional ::V_real(:)
     complex(SP), optional ::V_complex(:)
     logical,     optional ::state_is_2do(:),warn_me
   end subroutine
   !
 end interface
 !
 contains
   !
   subroutine QP_XCo_alloc(l_Vxc,l_Vnl_xc,l_screen_hyb)
   implicit none
     logical, optional, intent(in) :: l_Vxc,l_Vnl_xc,l_screen_hyb
     logical :: l_Vxc_,l_Vnl_xc_,l_screen_hyb_
     !
     l_Vxc_=.false.
     l_Vnl_xc_=.false.
     l_screen_hyb_=.false.
     if (present(l_Vxc)) l_Vxc_=l_Vxc
     if (present(l_Vnl_xc)) l_Vnl_xc_=l_Vnl_xc
     if (present(l_screen_hyb)) l_screen_hyb_=l_screen_hyb
     !
     if (.not.allocated(QP_Vxc).and.l_Vxc_) then
       YAMBO_ALLOC(QP_Vxc,(QP_n_states))
     endif
     if (.not.allocated(QP_Vnl_xc).and.l_Vnl_xc_) then
       YAMBO_ALLOC(QP_Vnl_xc,(QP_n_states))
     endif
     if (.not.allocated(QP_Vnl_sex).and.l_screen_hyb_) then
       YAMBO_ALLOC(QP_Vnl_sex,(QP_n_states))
     endif
     if (l_Vnl_xc_)     QP_Vnl_xc=cZERO
     if (l_Vxc_)        QP_Vxc=cZERO
     if (l_screen_hyb_) QP_Vnl_sex=cZERO
   end subroutine 
   !
   subroutine QP_XCo_free(l_Vxc,l_Vnl_xc,l_screen_hyb)
   implicit none
     logical, optional, intent(in) :: l_Vxc,l_Vnl_xc,l_screen_hyb
     logical :: l_Vxc_,l_Vnl_xc_,l_screen_hyb_
     !
     l_Vxc_=.false.
     l_Vnl_xc_=.false.
     l_screen_hyb_=.false.
     if (present(l_Vxc)) l_Vxc_=l_Vxc
     if (present(l_Vnl_xc)) l_Vnl_xc_=l_Vnl_xc
     if (present(l_screen_hyb)) l_screen_hyb_=l_screen_hyb
     !
     if (l_Vxc_) then
       YAMBO_FREE(QP_Vxc)
     endif
     if (l_Vnl_xc_) then
       YAMBO_FREE(QP_Vnl_xc)
     endif
     if (l_screen_hyb_) then
       YAMBO_FREE(QP_Vnl_sex)
     endif
     !
   end subroutine QP_XCo_free
   !
   ! Lifetimes definition
   !----------------------
   real(SP) function Time_of(E,T_unit)
     use pars,      ONLY:zero_dfl
     use units,     ONLY:HBAR_eVfs,HA2EV
     use functions, ONLY:NAN
     real(SP)    :: E,G_evm1
     character(*):: T_unit
     !
     G_evm1=1._SP/abs(E)/HA2EV
     if ( NAN(G_evm1)                      ) G_evm1=0._SP
     if ( abs(E)*HA2EV*1000.<epsilon(1._SP)) G_evm1=0._SP
     Time_of=G_evm1*HBAR_eVfs
     ! 
     select case (T_unit) 
        case("as")     ! 10^-18 s
          Time_of=Time_of*1.E3_SP
        case("fs")     ! 10^-15 s
        case("ps")     ! 10^-12 s
          Time_of=Time_of/1.E3_SP
        case("ns")     ! 10^-9 s
          Time_of=Time_of/1.E6_SP
        case("mus")    ! 10^-6 s
          Time_of=Time_of/1.E9_SP
        case("millis") ! 10^-3 s
          Time_of=Time_of/1.E12_SP
        case("s")      ! s
          Time_of=Time_of/1.E15_SP
     end select
     !
   end function Time_of
   !
   ! Time-Ordering
   !----------------
   !
   subroutine QP_SET_temperature_pre_factor( )
     use drivers,  ONLY:Finite_Tel
     QP_time_order_sign=-1                ! T-order
     if (Finite_Tel) QP_time_order_sign=1 ! Retared
   end subroutine
   !
   subroutine QP_copy(qp_in,qp,what,OP,pre_factor)
     use pars,          ONLY:SP,cI,rONE,rZERO
     use electrons,     ONLY:n_sp_pol
     use descriptors,   ONLY:IO_desc_duplicate
     type(QP_t)   ::qp_in,qp
     character(*) ::what,OP
     real(SP)     ::pre_factor,operation
     operation =1.
     if (OP=="-") operation =-1.
     qp%nk=qp_in%nk
     qp%nb=qp%nb
     qp%n_states=qp_in%n_states
     qp%GreenF_n_steps=qp_in%GreenF_n_steps
     if (what=="ALL") call IO_desc_duplicate(qp_in%desc,qp%desc)
     YAMBO_ALLOC(qp%k,(qp%nk,3))
     YAMBO_ALLOC(qp%table,(qp%n_states,3+n_sp_pol-1))
     YAMBO_ALLOC(qp%wk,(qp%nk))
     qp%k=qp_in%k
     qp%table=qp_in%table
     qp%wk=qp_in%wk
     if (allocated(qp_in%E)) then
       YAMBO_ALLOC(qp%Z,(qp%n_states))
       YAMBO_ALLOC(qp%E,(qp%n_states))
       YAMBO_ALLOC(qp%E_bare,(qp%n_states))
       qp%Z=rONE
       qp%E=rZERO
       qp%E_bare=qp_in%E_bare
       if (what=="ALL".or.index(what,"Z")>0) then
         qp%Z=qp_in%Z
       endif
       if (what=="ALL".or.index(what,"E")>0) then
         qp%E=qp%E_bare+operation*pre_factor*real(qp_in%E-qp_in%E_bare)
       endif
       if (what=="ALL".or.index(what,"W")>0) then
         qp%E=qp%E+operation*pre_factor*cI*aimag(qp_in%E-qp_in%E_bare)
       endif
     endif
     if (allocated(qp_in%S_total)) then
       YAMBO_ALLOC(qp%S_total,(qp%n_states,qp%GreenF_n_steps))
       YAMBO_ALLOC(qp%GreenF,(qp%n_states,qp%GreenF_n_steps))
       YAMBO_ALLOC(qp%GreenF_W,(qp%n_states,qp%GreenF_n_steps))
       qp%S_total=qp_in%S_total  
       qp%GreenF=qp_in%GreenF  
       qp%GreenF_W=qp_in%GreenF_W  
     endif
   end subroutine
   !
   subroutine QP_alloc(qp)
     use electrons, ONLY:n_sp_pol
     type(QP_t)::qp
     YAMBO_ALLOC(qp%k,(qp%nk,3))
     YAMBO_ALLOC(qp%table,(qp%n_states,3+n_sp_pol-1))
     YAMBO_ALLOC(qp%wk,(qp%nk))
     YAMBO_ALLOC(qp%Z,(qp%n_states))
     YAMBO_ALLOC(qp%E,(qp%n_states))
     YAMBO_ALLOC(qp%E_bare,(qp%n_states))
     if (qp%GreenF_n_steps>0) then
       YAMBO_ALLOC(qp%S_total,(qp%n_states,qp%GreenF_n_steps))
       YAMBO_ALLOC(qp%GreenF,(qp%n_states,qp%GreenF_n_steps))
       YAMBO_ALLOC(qp%GreenF_W,(qp%n_states,qp%GreenF_n_steps))
     endif
   end subroutine
   !
   subroutine QP_reset(qp)
     type(QP_t)::qp
     qp%nk=0
     qp%nb=0
     qp%n_states=0
     qp%GreenF_n_steps=0
     call IO_desc_reset(qp%desc)
     YAMBO_FREE(qp%table)
     YAMBO_FREE(qp%k)
     YAMBO_FREE(qp%wk)
     YAMBO_FREE(qp%E)
     YAMBO_FREE(qp%Z)
     YAMBO_FREE(qp%E_bare)
     YAMBO_FREE(qp%S_total)
     YAMBO_FREE(qp%GreenF)
     YAMBO_FREE(qp%GreenF_W)
   end subroutine
   !
   ! Lifetime functions e2et/h2ht
   !------------------------------
   !
   ! Gamma_n = 2 i \sum_m  { -/+ i Im[e^-1(e_n -e_m) (spin_occ-f+bose_f) <- e2et
   !                           + i Im[e^-1(e_m -e_n) (         f+bose_f) <- h2ht }
   !
   ! where - for T-ordered theory, + for causal (finite Tel)
   !
   function e2et(is,os,E,F)
     !
     use functions,   ONLY:bose_f
     use electrons,   ONLY:levels
     use electrons,   ONLY:spin_occ
     integer      ::is(3),os(3),e2et
     type(levels) ::E
     real(SP) :: F
     real(SP) :: dE !ws
     e2et=0
     !
     ! "Electron 2 Electron" decay
     !
     dE=E%E(is(1),is(2),is(3))-E%E(os(1),os(2),os(3))
     !
     F=QP_time_order_sign*(spin_occ-E%f(os(1),os(2),os(3))+bose_f(dE))
     !
     if (dE>0..and.abs(F)>epsilon(1.)) e2et=1
     if (e2et==0) F=0.
   end function
   !
   function h2ht(is,os,E,F)
     !
     use functions, ONLY:bose_f
     use electrons, ONLY:levels
     integer      ::is(3),os(3),h2ht
     type(levels) ::E
     real(SP)     ::F
     !
     ! Work Space
     !
     real(SP) :: dE 
     h2ht=0
     !
     !"Hole 2 Hole" decay
     !
     dE=E%E(os(1),os(2),os(3))-E%E(is(1),is(2),is(3))
     !
     F=E%f(os(1),os(2),os(3))+bose_f(dE)
     !
     if (dE>0..and.abs(F)>epsilon(1.)) h2ht=1
     if (h2ht==0) F=0.
   end function
   !
end module 
