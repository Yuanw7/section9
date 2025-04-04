!
! License-Identifier: GPL
!
! Copyright (C) 2006 The Yambo Team
!
! Authors (see AUTHORS file for details): AM DS
!
module com
 !
 use pars,          ONLY:DP,SP,lchlen,schlen,max_n_ofs,max_open_ofs,n_max_jobstr,msg_len
 use com_interfcs,  ONLY:msg_characters_vector_int,msg_integers_vector_int,msg_reals_vector_int
 !
 implicit none
 !
 ! PATHS
 !
 character(lchlen)  :: core_io_path
 character(lchlen)  :: more_io_path
 character(lchlen)  :: com_path
 !
 character(lchlen)  :: repfile
 !
 integer            :: repfile_unit
 integer            :: inpfile_unit
 !
 ! Path for the shifted grids
 !
 character(lchlen) :: grid_paths
 !
 character(3)      :: terminator(4)
 !
 ! JOB strings
 !
 integer            :: n_alt_jobstr
 character(lchlen)  :: jobstr
 character(lchlen)  :: alt_jobstr(n_max_jobstr)
 character(lchlen)  :: jobdir
 character(lchlen)  :: alt_jobdir(n_max_jobstr)
 !
 ! Sectioning
 !
 integer           :: depth,isec(5)
 integer           :: depth_SAVE_pp,isec_SAVE_pp(5) ! +/-P
 integer           :: depth_SAVE_p,isec_SAVE_p(5)   ! p
 character(2)      :: previous_secmode
 character(schlen) :: secnm
 !
 ! Output Files/Report File
 !
 character(lchlen)  :: of_name(max_n_ofs)
 character(lchlen)  :: of_opened(max_open_ofs)
 integer            :: of_unit(max_open_ofs+2)
 integer            :: n_ofs_defined
#if defined _NC_OUTPUT
 integer            :: of_nc_IDs(max_open_ofs)
 integer            :: of_nc_pos(4,max_open_ofs)
 character(schlen)  :: of_nc_var_names(max_open_ofs)
#endif
#if defined _YAML_OUTPUT
 integer            :: depth_yaml
 integer            :: of_yaml_IDs(max_open_ofs+2)
#endif
 !
 logical            :: write_to_of
 logical            :: write_to_report
 logical            :: fat_log
 logical            :: exp_user
 !
 interface
   !
   ! This interfaces is not really needed and could be easely
   ! dropped splitting of_open_close in two different subroutines:
   ! of_open and of_close
   !
   subroutine of_open_close(of_name,mode)
     character(*), intent(in)            :: of_name
     character(*), intent(in), optional  :: mode
   end subroutine of_open_close
   !
   function com_compose_msg(pre,body,post)
     use pars,          ONLY:msg_len
     character(msg_len)                  :: com_compose_msg
     character(*), intent(in)            :: body
     character(*), intent(in), optional  :: pre,post
   end function com_compose_msg
   !
 end interface
 !
 interface msg
   module procedure msg_novalue,                        &
   &                msg_logical,                        &
   &                msg_character,msg_characters_vector,&
   &                msg_integer,  msg_integers_vector,  &
   &                msg_real,     msg_reals_vector
 end interface
 !
 contains
   !
   !#######################
   ! INTERFACE TO MSG
   !#######################
   !
   subroutine msg_novalue(how,mesg)
     character(*)     :: mesg
     character(*)     :: how
     character        :: crap(1)
     integer          :: INDENT
     crap(1)=" "
     INDENT=-1
     if(len_trim(mesg)==0) then
       call msg_characters_vector(how," ",crap,INDENT=INDENT)
       return
     endif
     if (mesg(1:1)=="#") INDENT=0
     call msg_characters_vector(how,trim(mesg),crap,INDENT=INDENT)
   end subroutine
   !
   subroutine msg_logical(how,mesg,val,after_msg,USE_TABS,INDENT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     logical          :: val
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     if (     val) call msg_character(how,mesg,"yes",USE_TABS=USE_TABS,INDENT=INDENT,after_msg=after_msg,MSG_string=MSG_string)
     if (.not.val) call msg_character(how,mesg,"no" ,USE_TABS=USE_TABS,INDENT=INDENT,after_msg=after_msg,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_character(how,mesg,val,after_msg,USE_TABS,INDENT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     character(*)     :: val
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     call msg_characters_vector_int(how,mesg,(/val/),USE_TABS=USE_TABS,INDENT=INDENT,after_msg=after_msg,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_integer(how,mesg,val,after_msg,USE_TABS,INDENT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     integer          :: val
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     call msg_integers_vector_int(how,mesg,(/val/),USE_TABS=USE_TABS,&
&                                 INDENT=INDENT,after_msg=after_msg,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_real(how,mesg,val,after_msg,USE_TABS,INDENT,EXT_FORMAT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     real(SP)         :: val
     character(*), optional :: after_msg
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     logical,optional :: EXT_FORMAT
     character(lchlen), optional :: MSG_string
     call msg_reals_vector_int(how,mesg,(/val/),after_msg=after_msg,USE_TABS=USE_TABS,&
&                              INDENT=INDENT,EXT_FORMAT=EXT_FORMAT,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_characters_vector(how,mesg,val,after_msg,USE_TABS,INDENT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     character(*)     :: val(:)
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     call msg_characters_vector_int(how,mesg,val,after_msg=after_msg,USE_TABS=USE_TABS,INDENT=INDENT,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_integers_vector(how,mesg,val,after_msg,USE_TABS,INDENT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     integer          :: val(:)
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     call msg_integers_vector_int(how,mesg,val,after_msg=after_msg,USE_TABS=USE_TABS,&
&                                 INDENT=INDENT,MSG_string=MSG_string)
   end subroutine
   !
   subroutine msg_reals_vector(how,mesg,val,after_msg,USE_TABS,INDENT,EXT_FORMAT,MSG_string)
     character(*)     :: mesg
     character(*)     :: how
     real(SP)         :: val(:)
     integer,optional :: INDENT
     logical,optional :: USE_TABS
     logical,optional :: EXT_FORMAT
     character(lchlen), optional :: MSG_string
     character(*),      optional :: after_msg
     call msg_reals_vector_int(how,mesg,(/val/),after_msg=after_msg,USE_TABS=USE_TABS,&
&                              INDENT=INDENT,EXT_FORMAT=EXT_FORMAT,MSG_string=MSG_string)
   end subroutine
   !
   character(msg_len) function COMPOSE_spin_msg(title,title_len,msg_,i_spin)
     !
     use pars, ONLY:schlen
     use electrons, ONLY:n_sp_pol
     !
     implicit none
     !
     character(*) :: msg_,title
     integer, intent(in) :: i_spin,title_len
     !ws
     character(schlen) :: spin_ch
     spin_ch=" "
     !
     if (n_sp_pol==2.and.i_spin==1) spin_ch=" [spin UP]"
     if (n_sp_pol==2.and.i_spin==2) spin_ch=" [spin DN]"
     COMPOSE_spin_msg=com_compose_msg(title(1:title_len),msg_,spin_ch)
     !
   end function
   !
end module com
