!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM
!
subroutine rim(mode)
 !
 use pars,          ONLY:SP,pi,DP
 use com,           ONLY:msg
 use parallel_m,    ONLY:PP_indexes,myid,PP_indexes_reset
 use parallel_int,  ONLY:PP_redux_wait,PARALLEL_index 
 use LIVE_t,        ONLY:live_timing
 use vec_operate,   ONLY:v_norm2
 use D_lattice,     ONLY:alat
 use R_lattice,     ONLY:RL_vol,k_grid_uc_vol,k_grid_b,nqbz,&
&                        nqibz,RIM_epsm1,RIM_is_diagonal,RIM_RL_vol,&
&                        RIM_n_rand_pts,RIM_ng,RIM_W_ng,RIM_qpg,RIM_W_is_diagonal,&
&                        RIM_id_epsm1_reference,RIM_anisotropy,RIM_W,&
&                        cut_is_slab,idir
 use timing_m,      ONLY:timing
 !
#include<memory.h>
 !
 character(1)       :: mode
 !
 ! Work Space
 !
 type(PP_indexes)::px
 integer   :: iq
 real(SP)  :: em1_anis(3),G_radii,G_circ
 !
 ! Random generator
 !
 real(SP)  :: first_neighbour(26,3),fn_dist(26)
 integer   :: N_out,N_in,N_out_G,rep_factor,inn1,inn2,inn3,ic
 real(SP)  :: v1(3),v2(3),v1_norm(2)
 real(SP), allocatable  :: qr(:,:)
 integer            :: iseed(8)
 real(DP), external :: dlaran
 ! 
 if (mode == "x") call timing('RIM',OPR='start')
 if (mode == "c") call timing('RIM-W',OPR='start')
 !
 call PP_indexes_reset(px)
 !
 ! Anisotropy Setup
 !
 ! em1_anis refers to the X part only of epsm1 as in the self-energy
 ! the bare part is embodied in the exchange.
 !
 YAMBO_ALLOC(qr,(3,RIM_n_rand_pts))
 !
 em1_anis=RIM_epsm1(:)-1.
 !
 if (RIM_id_epsm1_reference<0.or.RIM_id_epsm1_reference>3) RIM_id_epsm1_reference=0
 if (RIM_id_epsm1_reference==0) em1_anis=0.
 if (RIM_id_epsm1_reference>0 .and. mode == "x") em1_anis=em1_anis/em1_anis(RIM_id_epsm1_reference)
 !
 ! Filling the small BZ with the random grid
 !===========================================
 !
 if (mode == "x") then
   !
   call section('+','RIM initialization')
   if (RIM_is_diagonal) call msg('r','* Diagonal components only detected *')
   !
 else 
   call section('=','RIM-W integrals')
   if (RIM_W_is_diagonal) call msg('r','* Diagonal components only detected *')
 end if
 !
 call msg('nr','8 x (sBL volume)        ',8._SP*k_grid_uc_vol,"[a.u.]")
 call msg('r', 'sBZ random points       ',RIM_n_rand_pts)
 !
 ! Random generator seed
 !
! call date_and_time(ch(1),ch(2),ch(3),iseed)
! iseed=iabs(iseed)
! ! iseed(4) must be odd
! iseed(4)=2*(iseed(4)/2)+1
 !
 ! AF: fixing the seed for the sake of
 !     data reproducibility.
 !
 iseed=0
 iseed(4:8) = (/5,6,7,8,9/)
 !
 ! First neighbours of the gamma point in the k_grid_b units
 !
 ic=0
 do inn1=-1,1
   do inn2=-1,1
    do inn3=-1,1
      if (all((/inn1,inn2,inn3/)==0)) cycle
      ic=ic+1
      first_neighbour(ic,:)=matmul(transpose(k_grid_b),(/inn1,inn2,inn3/))
     enddo
   enddo
 enddo
 ! Precompute the neighbours distances
 do inn1=1,26
  fn_dist(inn1)=v_norm2(first_neighbour(inn1,:))  
 end do
 !
 ! Loop setup
 !
 N_in=1
 N_out=0
 rep_factor=RIM_n_rand_pts/100
 if (rep_factor==0) rep_factor=1
 !
 call live_timing('Random points',RIM_n_rand_pts/rep_factor)
 loop: do while(.not.N_in==RIM_n_rand_pts+1)
   !
   do ic=1,3
     v2(ic)=2.*dlaran(iseed(4:))-1
   enddo
   N_out=N_out+1
   !
   ! From rlu in the k_grid_b units (v2) to Cartesian (v1)
   !
   v1=matmul(transpose(k_grid_b),v2) 
   !v1_norm(1)=v_norm2(v1)
   do inn1=1,26
     v1_norm(2)=(v1(1)* first_neighbour(inn1,1) &
                +v1(2)* first_neighbour(inn1,2) &
                +v1(3)* first_neighbour(inn1,3))*2._SP
     if (fn_dist(inn1)<v1_norm(2))  cycle loop
     !v1_norm(2)=v_norm2(v1-first_neighbour(inn1,:))
     !if (v1_norm(2)<v1_norm(1)) cycle loop
     !if (v1_norm(2)<v1_norm(1)-1.E-5) cycle loop
     !
     ! The small number 1E-5 seems unessential and bring to erroneous integration in
     ! the case of large samplings.
     ! 
   enddo
   qr(:,N_in)=v1*alat(:)/2._SP/pi
   N_in=N_in+1
   if (mod(N_in,rep_factor)==0) call live_timing(steps=1)
 enddo loop
 !
 !Only the 2D-BZ is sampled for 2D systems
 if (cut_is_slab) qr(idir(1),:) = 0._SP
 !
 call live_timing()
 call msg('r','Points outside the sBZ  ',N_out)
 !  
 !Integrated RL VOLUME 
 !
 RIM_RL_vol=8._SP*k_grid_uc_vol*real(RIM_n_rand_pts)/real(N_out)*real(nqbz)
 call msg('r', 'RL volume               ',RL_vol,"[a.u.]")
 call msg('rn','Integrated volume       ',RIM_RL_vol,"[a.u.]")
 !
 call section('=','RIM integrals')
 !
 if (mode == "x") then
   YAMBO_ALLOC(RIM_qpg,(nqibz,RIM_ng,RIM_ng))
   RIM_qpg=0._SP
   G_radii=0._SP
   G_circ=0._SP
   N_out_G=0
 else if (mode == "c") then
   YAMBO_ALLOC(RIM_W,(nqibz,RIM_W_ng,RIM_W_ng))
   RIM_W=0._SP
 end if
 !
 call PARALLEL_index(px,(/nqibz/))
 call live_timing('Momenta loop',px%n_of_elements(myid+1))
 !
 do iq=1,nqibz
   if (.not.px%element_1D(iq)) cycle
   !
   if (mode == "x") then
     call rim_integrate_v(iq,qr,em1_anis,N_out,N_out_G,G_radii,G_circ)
   else
     call rim_integrate_w(iq,qr,N_out,em1_anis)
   endif
   call live_timing(steps=1)
   !
 enddo
 !
 call live_timing()
 !
 if (mode == "x") call PP_redux_wait(RIM_qpg)
 if (mode == "c") call PP_redux_wait(RIM_W)
 !
 call PP_redux_wait(RIM_anisotropy)
 !
 if (mode == "x") then
   !
   call PP_redux_wait(G_radii)
   call PP_redux_wait(N_out_G)
   !
   call msg('r','Gamma point sphere radius             ',G_radii,"[a.u.]")
   call msg('r','Points outside the sphere             ',N_out_G)
   call msg('r','[Int_sBZ(q=0) 1/q^2]*(Vol_sBZ)^(-1/3) =',&
&               4._SP*pi**3*RIM_qpg(1,1,1)*k_grid_uc_vol**(-1._SP/3._SP))
   call msg('r','                               should be <',7.7956_SP)
   if (RIM_id_epsm1_reference/=0) then
     call msg('r','Anisotropy correction            [o/o]',&
&               (RIM_anisotropy-RIM_qpg(1,1,1))/RIM_qpg(1,1,1)*100._SP)
   endif
   !
 endif
 !
 ! CLEAN
 !
 call PP_indexes_reset(px)
 YAMBO_FREE(qr)
 !
 if (mode == "x") call timing('RIM',OPR='stop')
 if (mode == "c") call timing('RIM-W',OPR='stop')
 !
end subroutine
