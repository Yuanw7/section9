!
! License-Identifier: GPL
!
! Copyright (C) 2015 The Yambo Team
!
! Authors (see AUTHORS file for details): DS AM AF IM
!
!> @callgraph
!> @callergraph
subroutine X_irredux_residuals(Xen,Xk,X,Dip,i_cg,iq,Xo_res,Xo_scatt)
 !
 ! This subroutine must be kept thread-safe, since it is used
 ! inside an external omp loop.
 ! This means mostly that variables from modules should be used as
 ! intent(IN) and not modified.
 !
 use pars,          ONLY:SP,cZERO,cONE
 use wrapper,       ONLY:V_plus_alpha_V
 use DIPOLES,       ONLY:DIPOLE_t,DIP_rotated
 use X_m,           ONLY:X_t,X_poles_tab,l_X_terminator,X_cols,X_rows,drude_n_states
 use electrons,     ONLY:levels,spin_occ
 use frequency,     ONLY:bare_grid_N,coarse_grid_Pt
 use collision_el,  ONLY:elemental_collision
 use D_lattice,     ONLY:nsym,DL_vol,i_time_rev,sop_inv
 use R_lattice,     ONLY:qindx_X,bz_samp,q0_def_norm
 use vec_operate,   ONLY:v_norm
 use collision_el,  ONLY:elemental_collision
 use deviceXlib_m
 !
#include<dev_defs.h>
 !
 use R_lattice,     ONLY:DEV_VAR(g_rot),DEV_VAR(G_m_G)
 !
 implicit none
 !
 type(elemental_collision), target :: Xo_scatt
 type(levels), intent(in) :: Xen
 type(bz_samp),intent(in) :: Xk
 type(X_t),    intent(in) :: X
 type(DIPOLE_t),intent(in):: Dip
 integer,      intent(in) :: i_cg,iq
 complex(SP),  intent(out) DEV_ATTR :: Xo_res(X_rows(1):X_rows(2),X_cols(1):X_cols(2))
 !
 ! Work sapce
 !
 ! AF: rhotw_save defined as automatic goes to stack and can harm 
 !     (stack overflow, causing random crashes of the code)
 !     it would be best to define it as allocatable (less efficient if too
 !     many calls are done, though)
 !
 complex(SP), pointer DEV_ATTR :: Xo_scatt_rhotw_p(:)
 complex(SP) DEV_ATTR :: DEV_VAR(rhotw_save)(Xo_scatt%ngrho),DEV_VAR(rhotw_save2)(Xo_scatt%ngrho)
 integer            :: sop_inv_l,X_ng
 complex(SP)        :: ctmp,Z_
 real(SP)           :: Z_eh_occ
 logical            :: l_X_term_vv
 integer            :: ngrho
 integer            :: ig1,ig2,ik,is,ikp,ikbz,ikpbz,i_spin,&
&                      isp,iv,ic,isave(4),n_poles,i_bg,ig_start,ROWS(2)
 !
 ! DIPOLES
 !
 complex(SP)        :: DIP_projected,field_dir(3)
 !
 field_dir=Dip%q0/v_norm(Dip%q0)*q0_def_norm
 !
 isave     = 0
 n_poles   = sum(bare_grid_N(1:i_cg-1))
 Xo_res    = cZERO
 Z_        = cONE
 !
 ngrho     = Xo_scatt%ngrho
 !
 loop_bare_grid: do i_bg = 1,bare_grid_N(i_cg)
   !
   n_poles=n_poles+1
   !
   ! Scattering geometry
   !---------------------
   !
   ikbz   = X_poles_tab(n_poles,1)
   iv     = X_poles_tab(n_poles,2)
   ic     = X_poles_tab(n_poles,3)
   i_spin = X_poles_tab(n_poles,4)
   !
   ikpbz  = qindx_X(iq,ikbz,1)
   !
   ik = Xk%sstar(ikbz,1)
   is = Xk%sstar(ikbz,2)
   !
   ikp= Xk%sstar(ikpbz,1)
   isp= Xk%sstar(ikpbz,2)
   !
   l_X_term_vv = (l_X_terminator.and.ic>=X%ib(1).and.ic<=Xen%nbm(i_spin))
   !
   ! take into account the cancellation of terms
   ! occurring when dealing with the wings
   ig_start=X_rows(1)
   if (iq==1) ig_start= max(2,ig_start)
   !
   ! Note the renormalization of the Z_eh_occ=f(1-f)*Z factor
   !
   if (allocated(Xen%Z))      Z_=Xen%Z(ic,ik,i_spin)*Xen%Z(iv,ikp,i_spin)
   !
   if (allocated(Xen%GreenF)) Z_=cONE
   !
   Z_eh_occ = Xen%f(iv,ikp,i_spin)*(spin_occ-Xen%f(ic,ik,i_spin))/spin_occ/real(Xk%nbz,SP)/DL_vol*real(Z_)
   !
   if (iq==1.and.abs(coarse_grid_Pt(i_cg))<Dip%Energy_treshold) Z_eh_occ=1._SP
   !
   if (l_X_term_vv) Z_eh_occ = Xen%f(iv,ikp,i_spin)*Xen%f(ic,ik,i_spin)/spin_occ/real(Xk%nbz,SP)/DL_vol*real(Z_)
   !
   ! Scattering CALL
   !-----------------
   !
   X_ng=X%ng
   Xo_scatt_rhotw_p => DEV_VAR(Xo_scatt%rhotw)
   !
   if (iq==1) then
     !
     ! iq=1: Symmetries are applyed later and
     !       iG=1 case is computed separately 
     !
     Xo_scatt%is = (/ic,ik,1,i_spin/)
     Xo_scatt%os = (/iv,ik,1,i_spin/)
     Xo_scatt%qs = (/1,1,1/)
     if (.not. X%ng==1) then
       if ( any((/isave(1)/=iv,isave(2)/=ic,isave(3)/=ik,isave(4)/=i_spin/)) ) then
         !
         call DEV_SUB(scatter_Bamp)(Xo_scatt)
         !
         ! rhotw_save=Xo_scatt%rhotw
         call dev_memcpy(DEV_VAR(rhotw_save),DEV_VAR(Xo_scatt%rhotw))
         !
         isave=(/iv,ic,ik,i_spin/)
         !
       endif
       !
       sop_inv_l=sop_inv(is)
       !
#ifdef _CUDA
       !$cuf kernel do(1) <<<*,*>>>
#else
       !$omp parallel do default(shared), private(ig1,ig2)
#endif
       do ig1=2,X_ng
         ig2=DEV_VAR(g_rot)(ig1,sop_inv_l)
         Xo_scatt_rhotw_p(ig1)=DEV_VAR(rhotw_save)(ig2)
       enddo
       !
       if (is>nsym/(i_time_rev+1)) then
         call dev_conjg( DEV_VAR(Xo_scatt%rhotw) )
       endif
       !
     endif
     !
     ! iG=1 case
     DIP_projected=dot_product(field_dir,DIP_rotated(ic,iv,ikbz,i_spin,"DIP_iR",Xk))
     if(iv/=ic) DEV_VAR(Xo_scatt%rhotw)(1)=    -conjg(DIP_projected)
     !
     ! 2022/05/11 DS & AF
     ! Note: lim (q->0) <psi_nk+q | e-^{-i*q*r} | psi_nk > = <psi_nk   | 1 | psi_nk > = 1._SP
     !       in the code the value cONE is needed for the Drude term;
     !       to be checked what to do with the terminator
     !       at present with the terminator we use 0.0_SP
     !
     ! If I take the expansion e-^{-i*q*r}= 1 + O(q) for the leading term this gives instead
     !       lim (q->0) <psi_nk+q | 1 | psi_nk > = 0._SP 
     !       Only using also <psi_nk+q | = <psi_nk | (1 + O(q) ) we recover the correct result.
     !
     if(iv==ic) DEV_VAR(Xo_scatt%rhotw)(1)=cONE-conjg(DIP_projected)
     !
     if (l_X_term_vv.and.iv==ic) DEV_VAR(Xo_scatt%rhotw)(1)=0.0_SP
     !
   else
     !
     ! iq>1: direct computation
     !
     Xo_scatt%is=(/ic,ik,is,i_spin/)
     Xo_scatt%os=(/iv,ikp,isp,i_spin/)
     Xo_scatt%qs=(/qindx_X(iq,ikbz,2),iq,1/)
     !
     call DEV_SUB(scatter_Bamp)(Xo_scatt)
     !
   endif
   !
   ! Filling the upper triangular part of the residual here ! 
   !-------------^^^^^---------------------------------------
#ifdef _CUDA
   !$cuf kernel do(2) <<<*,*>>>
   do ig2=X_cols(1),X_cols(2)
     do ig1=X_rows(1),X_rows(2)
       if (ig1 <= ig2) then
         Xo_res(ig1,ig2) = Xo_res(ig1,ig2) + Z_eh_occ*Xo_scatt_rhotw_p(ig2) &
&                                          * conjg(Xo_scatt_rhotw_p(ig1))
       endif
     enddo
   enddo
#else
   !$omp parallel do default(shared), private(ig2)
   do ig2=X_cols(1),X_cols(2)
     ROWS=(/X_rows(1),min(ig2,X_rows(2))/)
     call V_plus_alpha_V(ROWS(2)-ROWS(1)+1,Z_eh_occ*Xo_scatt%rhotw(ig2),&
&                       conjg(Xo_scatt%rhotw(ROWS(1):ROWS(2))),Xo_res(ROWS(1):ROWS(2),ig2))
   enddo
   !$omp end parallel do
#endif
   !
   ! add terminator specific corrections
   ! (ic is running in valence)
   !
   if (l_X_term_vv.and.iv==ic) then 
     !
     Xo_scatt%is = (/iv,ik,1,i_spin/)
     Xo_scatt%os = (/iv,ik,1,i_spin/)
     Xo_scatt%qs = (/1,1,1/)
     !
     call dev_memcpy(DEV_VAR(rhotw_save2),DEV_VAR(Xo_scatt%rhotw))
     !
     if (X%ng==1) then
       DEV_VAR(Xo_scatt%rhotw)(1)=cONE 
     else
       call DEV_SUB(scatter_Bamp)(Xo_scatt)
     endif
     !
     ! symm
     ! rhotw_save=Xo_scatt%rhotw
     call dev_memcpy(DEV_VAR(rhotw_save),DEV_VAR(Xo_scatt%rhotw))
     !
     sop_inv_l=sop_inv(is)
     !
#ifdef _CUDA
     !$cuf kernel do(1) <<<*,*>>>
#else
     !$omp parallel do default(shared), private(ig1,ig2)
#endif
     do ig1=1,X_ng
       ig2=DEV_VAR(g_rot)(ig1,sop_inv_l)
       Xo_scatt_rhotw_p(ig1)=DEV_VAR(rhotw_save)(ig2)
     enddo
     !
     if (is>nsym/(i_time_rev+1)) then
       call dev_conjg(DEV_VAR(Xo_scatt%rhotw))
     endif
     !
#ifdef _CUDA
     !$cuf kernel do(2) <<<*,*>>>
     do ig2=X_cols(1),X_cols(2)
       do ig1=ig_start,X_rows(2)
         if (ig1 > ig2) cycle
         Xo_res(ig1,ig2)=Xo_res(ig1,ig2)-Z_eh_occ*Xo_scatt_rhotw_p(G_m_G_d(ig2,ig1))
       enddo
     enddo
#else
     !$omp parallel do default(shared), private(ig1,ig2)
     do ig2=X_cols(1),X_cols(2)
       do ig1=ig_start,min(ig2,X_rows(2))
         Xo_res(ig1,ig2)=Xo_res(ig1,ig2)-Z_eh_occ*Xo_scatt%rhotw(G_m_G(ig2,ig1))
       enddo
     enddo
#endif
     !
     call dev_memcpy(DEV_VAR(Xo_scatt%rhotw),DEV_VAR(rhotw_save2))
     !
   endif
   !
 enddo loop_bare_grid
 !
 if (iq==1.and.abs(coarse_grid_Pt(i_cg))<Dip%Energy_treshold) then
    ctmp=Xo_res(1,1)
    Xo_res(1,1)=ctmp/real(drude_n_states,SP)
 endif
 !
end subroutine X_irredux_residuals
