!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM DS
!
subroutine FREQUENCIES_setup(W)
 !
 ! %er %dr %n_freqs -> %p
 !
 use pars,          ONLY:SP,cI
 use frequency,     ONLY:w_samp
 use parser_m,      ONLY:parser
#include<memory.h>
 !
 type(w_samp):: W 
 ! 
 ! Work Space
 !
 integer          :: i1
 logical          :: metallic_damping
 real(SP)         :: delta_w,w_n
 complex(SP)      :: w_tmp,w_tmp2(2)
 real(SP),external:: FREQUENCIES_damping
 !
 !If %p is associated and consitent with %er/%dr return
 !
 if (allocated(W%p)) then
   if (size(W%p)==W%n_freqs.and.&
&      abs(  real(W%p(1))-W%er(1) )<1.E-5.and.&
&      abs( aimag(W%p(1))-W%dr(1) )<1.E-5.and.&
&      abs(  real(W%p(W%n_freqs))-W%er(2) )<1.E-5.and.&
&      abs( aimag(W%p(W%n_freqs))-W%dr(2) )<1.E-5) return
 endif
 !
 YAMBO_FREE(W%p)

 !
 ! Build mpa-type grids
 !
 if (trim(W%grid_type)=="mpa-lP") then
   !
   if (mod(W%n_freqs,2)/=0) call error("[MPA] Invalid number of frequencies")
   YAMBO_ALLOC(W%p,(W%n_freqs))
   !
   w_tmp2(1:2)=cmplx(W%er(1:2),W%ir(1:2),kind=SP)
   call FREQUENCIES_mpa_sampling(W%n_freqs/2,w_tmp2(1),w_tmp2(2),W%dr,W%samp_type,'lP',W%p)
   !
   return
 endif

 !
 ! Regular grids
 ! First the total/mem steps
 !
 if (W%er(2)==W%er(1)) W%n_freqs=1
 W%n_freqs=nint(W%per_memstps*real(W%n_freqs)/100._SP)
 !
 YAMBO_ALLOC(W%p,(W%n_freqs))
 !
 W%p(1)=cmplx(W%er(1),W%dr(1),SP)
 if (W%n_freqs==1) then
   W%er(2)=W%er(1)
   W%dr(2)=W%dr(1)
   return
 endif
 !
 delta_w=(W%er(2)-W%er(1))/(W%n_freqs-1)
 !
 forall(i1=1:W%n_freqs-1) W%p(i1+1)=cmplx(W%er(1)+delta_w*real(i1,SP),0.,SP)
 !
 do i1=1,W%n_freqs
   W%p(i1)=real(W%p(i1))+FREQUENCIES_damping(W,real(W%p(i1)))*cI
   if(abs(W%p(i1))==0._SP) W%p(i1)=W%p(i1)+1.E-10_SP
 enddo
 !
 call parser('MetDamp',metallic_damping)
 !
 if(.not.metallic_damping) return
 !
 ! Davide
 ! Implementation of eq. 10, PRB 82, 035104 (2010): w=sqrt(Re[w]*(Re[w]+2*i*Im[w])),
 ! notice that this is sqrt(w^2+Im[w]^2)
 ! 1) This expression should guarantee the correct behaviour of Re[eps^m1] nearby w=0
 !    in metals. Sistems with a gap are not affected.
 ! 2) Thanks to this expression the length and the velocity gauge are finally identical
 !    and the divergency for w->0 in the velocity is removed (there is just some numerical noise)
 !
 do i1=1,W%n_freqs
   w_n=W%er(1)+delta_w*real(i1-1,SP)
   if(w_n>0._SP) W%p(i1)=cmplx( w_n**2/sqrt(w_n**2+aimag(W%p(i1))**2),aimag(W%p(i1)),SP)
   if(w_n<0._SP) W%p(i1)=cmplx(-w_n**2/sqrt(w_n**2+aimag(W%p(i1))**2),aimag(W%p(i1)),SP)
 enddo
 !
 do i1=1,W%n_freqs
   w_tmp=W%p(i1)
   if ( real(w_tmp)>=0._SP ) W%p(i1)= sqrt(real(w_tmp)*(real(w_tmp)+2._SP*aimag(w_tmp)*cI))
   if ( real(w_tmp)< 0._SP ) W%p(i1)=-sqrt(real(w_tmp)*(real(w_tmp)+2._SP*aimag(w_tmp)*cI))
   if (abs(W%p(i1))==0._SP ) W%p(i1)= W%p(i1)+1.E-10_SP
 enddo
 !
end subroutine
