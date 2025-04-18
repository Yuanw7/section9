!
! License-Identifier: GPL
!
! Copyright (C) 2017 The Yambo Team
!
! Authors (see AUTHORS file for details): AM DS
!
integer function io_connect(desc,subfolder,type,no_full_RESET,FORCE_READ_MODE,ID)
 !
 use pars,          ONLY:schlen,lchlen,num_of_alternative_locations
 use stderr,        ONLY:STRING_pack
 use parallel_m,    ONLY:ncpu,mpi_comm_null,myid
 use IO_int,        ONLY:get_name,all_locations
 use IO_m,          ONLY:read_is_on,write_is_on,open_is_on,io_reset,io_file,io_unit,io_type,&
&                        io_raw_extension,mk_dir,io_extension,io_folder,netcdf_call,&
&                        io_PAR_cpu,IO_NO_DATABASE,IO_CREATE_MODE,IO_READ_MODE,IO_WRITE_MODE,&
&                        IO_NOT_ALLOWED,OP_APP,OP_APP_CL,io_action,OP_WR,OP_WR_CL,io_PAR_comm,io_sec,io_mode
#if defined _PAR_IO && defined _HDF5_IO
 use IO_m,          ONLY:PAR_IO_CREATE_MODE,PAR_IO_WRITE_MODE
 use parallel_m,    ONLY:MPI_INFO_NULL
#endif
 use netcdf,        ONLY:nf90_open,nf90_create,nf90_enddef
#if defined _PAR_IO && defined _HDF5_IO
 use netcdf,        ONLY:nf90_open_par,nf90_create_par,nf90_nowrite
#endif
 use com,           ONLY:jobstr,more_io_path,n_alt_jobstr,jobdir
 !
 implicit none
 !
 character(*)           :: desc
 integer     , optional :: type,ID
 character(*), optional :: subfolder
 logical,      optional :: no_full_RESET
 logical,      optional :: FORCE_READ_MODE
 !
 ! Work Space
 !
 integer          ::i_jobstr
 character(lchlen)::alternative_file(num_of_alternative_locations),folder
 character(schlen)::subfolder_
 logical          ::file_found,full_RESET,par_IO,write_mode,read_mode
 !
 logical,           external :: file_exists
 !
 full_RESET=.true.
 if (present(no_full_RESET)) full_RESET=.not.no_full_RESET
 !
 write_mode=write_is_on(ID)
 read_mode =read_is_on(ID)
 if (present(FORCE_READ_MODE)) then
   if (FORCE_READ_MODE) then
     write_mode=.false.
     read_mode =.true.
   endif
 endif
 !
 io_connect=0
 if (.not.io_PAR_cpu(ID)==1) then
   io_connect=IO_NOT_ALLOWED
   return
 endif
 !
 ! Build the file name
 !
 subfolder_=' '
 if (present(subfolder)) subfolder_=subfolder
 !
 alternative_file=" "
 !
 if (len_trim(io_file(ID))==0) then
   io_file(ID)=get_name(desc,subfolder_,type,CORE_IO=.FALSE.,MORE_IO=.TRUE.,COM_IO=.FALSE.)
   if (read_mode .and. (type==1.or.type==2) ) alternative_file=all_locations(desc,subfolder_)
 endif
 !
 ! Keep the extension for the fragment-related procedures
 !
 if (len_trim(io_raw_extension(ID))==0) io_raw_extension(ID)=desc
 io_extension(ID) =desc
 io_folder(ID)    =trim(subfolder_)
 !
 ! Create Directories 
 !
 write (folder,'(2a)') trim(more_io_path),'/SAVE'
 if (len_trim(jobstr)>0) then
   if (len_trim(jobdir)> 0) write (folder,'(5a)') trim(more_io_path),'/',trim(jobdir),'/',trim(jobstr)
   if (len_trim(jobdir)==0) write (folder,'(3a)') trim(more_io_path),'/',trim(jobstr)
 endif
 !
 if (write_mode.and.(type==2.or.len_trim(jobstr)==0)) then
   call IO_make_directories("MORE")
   call mk_dir(folder)
   if (present(subfolder)) call mk_dir(STRING_pack(folder,"/",subfolder))
 endif
 !
 if (.not.open_is_on(ID)) return
 !
 ! If to write
 !
 if (write_mode) then
   !
   par_IO=io_PAR_comm(ID)/=mpi_comm_null.and.ncpu>1
   !
   ! DEBUG <
   !write(*,*) myid,"writing to file ",trim(io_file(ID)),file_exists(trim(io_file(ID))),par_IO
   ! DEBUG >
   !
   if ( (io_action(ID)==OP_APP_CL.or.io_action(ID)==OP_APP) .and. file_exists(trim(io_file(ID))) ) then
     !
     if(.not.par_IO) call netcdf_call(nf90_open(trim(io_file(ID)),IO_WRITE_MODE,io_unit(ID)),ID)
#if defined _PAR_IO && defined _HDF5_IO
     if(     par_IO) call netcdf_call(nf90_open_par(trim(io_file(ID)),PAR_IO_WRITE_MODE, &
&                                     io_PAR_comm(ID),MPI_INFO_NULL,io_unit(ID)),ID)
#else
     if(     par_IO) call error(' Parallel COMM present in io_connect, but yambo compiled with serial IO')
#endif
     !
   else
     !
     if(.not.par_IO) call netcdf_call(nf90_create(trim(io_file(ID)),IO_CREATE_MODE,io_unit(ID)),ID)
#if defined _PAR_IO && defined _HDF5_IO
     if(     par_IO) call netcdf_call(nf90_create_par(trim(io_file(ID)),PAR_IO_CREATE_MODE,&
&                                     io_PAR_comm(ID),MPI_INFO_NULL,io_unit(ID)),ID)
#else
     if(     par_IO) call error(' Parallel COMM present in io_connect, but yambo compiled with serial IO')
#endif
     call netcdf_call(nf90_enddef(io_unit(ID)),ID)
     !
     if (io_action(ID)==OP_APP_CL) io_action(ID)=OP_WR_CL
     if (io_action(ID)==OP_APP)    io_action(ID)=OP_WR
     !
   endif
   !
 endif
 !
 ! If to read
 !
 if (read_mode) then
   !
   if (.not.file_exists(trim(io_file(ID)))) then
     !
     file_found=.FALSE.
     !
     do i_jobstr=1,4+2*n_alt_jobstr
       file_found=file_exists(trim(alternative_file(i_jobstr)))
       if(file_found) exit
     enddo
     !
     if (.not.file_found) then
       io_connect=IO_NO_DATABASE
       if (.not.full_RESET) io_file(ID)=''
       if (     full_RESET) then
         call io_reset(ID)
         io_type(ID)=0
       endif
       return
     endif
     !
     io_file(ID)=alternative_file(i_jobstr)
     !
   endif
   !
   if (type==-2.or. type==-4) open(unit=io_unit(ID),file=trim(io_file(ID)),form='unformatted')
   if (type/=-2.and.type/=-4) call netcdf_call(nf90_open(trim(io_file(ID)),IO_READ_MODE,io_unit(ID)),ID)
   !
 endif
 !
 io_type(ID)=type
 !
end function io_connect
