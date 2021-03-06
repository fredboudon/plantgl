/* -*-c++-*-
 *  ----------------------------------------------------------------------------
 *
 *       PlantGL: The Plant Graphic Library
 *
 *       Copyright CIRAD/INRIA/INRA
 *
 *       File author(s): F. Boudon (frederic.boudon@cirad.fr) et al. 
 *
 *  ----------------------------------------------------------------------------
 *
 *   This software is governed by the CeCILL-C license under French law and
 *   abiding by the rules of distribution of free software.  You can  use, 
 *   modify and/ or redistribute the software under the terms of the CeCILL-C
 *   license as circulated by CEA, CNRS and INRIA at the following URL
 *   "http://www.cecill.info". 
 *
 *   As a counterpart to the access to the source code and  rights to copy,
 *   modify and redistribute granted by the license, users are provided only
 *   with a limited warranty  and the software's author,  the holder of the
 *   economic rights,  and the successive licensors  have only  limited
 *   liability. 
 *       
 *   In this respect, the user's attention is drawn to the risks associated
 *   with loading,  using,  modifying and/or developing or reproducing the
 *   software by the user in light of its specific status of free software,
 *   that may mean  that it is complicated to manipulate,  and  that  also
 *   therefore means  that it is reserved for developers  and  experienced
 *   professionals having in-depth computer knowledge. Users are therefore
 *   encouraged to load and test the software's suitability as regards their
 *   requirements in conditions enabling the security of their systems and/or 
 *   data to be ensured and,  more generally, to use and operate it in the 
 *   same conditions as regards security. 
 *
 *   The fact that you are presently reading this means that you have had
 *   knowledge of the CeCILL-C license and that you accept its terms.
 *
 *  ----------------------------------------------------------------------------
 */





#include "util_enviro.h"
#include "util_string.h"
#include "dirnames.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#ifndef PGL_CORE_WITHOUT_QT
#include <QtCore>
#endif


#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#include <winreg.h>
#include <LMCONS.H>
#define MAXPATHLEN _MAX_PATH

#else

#ifdef __GNUC__
#include <sys/param.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#endif

#ifdef _MSC_VER
#define STRING2(val) #val
#define STRING(val) STRING2(val)
#pragma message("MSVC VERSION: " STRING(_MSC_FULL_VER))
#ifdef _MSVC_LANG
#pragma message("C++  VERSION: " STRING(_MSVC_LANG))
#endif
#ifdef _WIN64
#pragma message("64bit ARCHITECTURE TARGETED")
#else
#ifdef _WIN32
#pragma message("32bit ARCHITECTURE TARGETED")
#else
#pragma message("NO ARCHITECTURE TARGETED")
#endif
#endif
#endif

using namespace std;

PGL_BEGIN_NAMESPACE

static string PLANTGL_DIR;
static string OPENALEA_DIR;

inline const char * pglgetenv(const char * name) {
    return getenv(name);
}

string getHome(){
        const char * home = pglgetenv("HOME");
#ifdef _WIN32
        if(!home)home = pglgetenv("USERPROFILE");
        if(!home)home = "C:\\";
#else
        if(!home)home = "/";
#endif
        return string(home);
}

string getOpenAleaDir(){
        if(!OPENALEA_DIR.empty())return OPENALEA_DIR;
        const char * dir = pglgetenv("OPENALEADIR");
        if(!dir)
#ifdef _WIN32
            dir = "C:\\openalea";
#else
            dir = "/usr/local/openalea";
#endif
        OPENALEA_DIR = string(dir);
        return OPENALEA_DIR;
}

string getPlantGLDir(){
    if(!PLANTGL_DIR.empty())return PLANTGL_DIR;
    const char * dir = pglgetenv("PLANTGLDIR");
    if(!dir)PLANTGL_DIR = getOpenAleaDir();
    else PLANTGL_DIR = string(dir);
    return PLANTGL_DIR;
}

void setPlantGLDir(const std::string& dir){
    PLANTGL_DIR = dir;
}

string getUserName(){
#if defined(_WIN32)
    char lpBuffer[UNLEN];
    DWORD nSize(UNLEN);
    if(GetUserNameA(lpBuffer,&nSize)){
         return string(lpBuffer);
    }
    else{
         return string("Windows User");
    }
#elif defined(__GNUC__)
    return string(getlogin());

#else
#pragma "message username not defined"
    return string("");
#endif
}

string getOSFamily(){
#ifdef SYSTEM_IS__Linux
        return string("Linux");
#elif SYSTEM_IS__IRIX
        return string("Irix");
#elif SYSTEM_IS__CYGWIN
        return string("Cygwin");
#elif _WIN32
        return string("Windows");
#elif _WIN64
        return string("Windows");
#elif defined(__APPLE__)
        return string("MacOSX");
#else
  #pragma "message OS Family not defined"
        return string("");
#endif
}

string getOSName(){
#ifdef _WIN32
        string sys_name;
        LPOSVERSIONINFO lpVersionInformation = new OSVERSIONINFO;
        lpVersionInformation->dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
        if(GetVersionEx(lpVersionInformation)){

                if(lpVersionInformation->dwPlatformId == VER_PLATFORM_WIN32s)
                        sys_name = "Windows 3.1";
                else if(lpVersionInformation->dwPlatformId == VER_PLATFORM_WIN32_WINDOWS)
                        sys_name = "Windows 9x";
                else if(lpVersionInformation->dwPlatformId == VER_PLATFORM_WIN32_NT)
                        sys_name = "Windows NT";

                sys_name += ' ' + number(lpVersionInformation->dwMajorVersion)
                                  + '.' + number(lpVersionInformation->dwMinorVersion) ;

                sys_name += " (";
                char a = lpVersionInformation->szCSDVersion[0];
                for(int i = 1 ; i < 128 && a != '\0'; i++){
                        sys_name += a;
                        a = lpVersionInformation->szCSDVersion[i];
                }
                sys_name += ")";
        }
        else sys_name = "Windows";
        return sys_name;
#elif defined (__GNUC__)
    struct utsname buf;
    uname(&buf);
    return string(buf.sysname);
#else
# pragma "message OS Name not defined"
    return string("");
#endif

}

string getOSVersion(){
#ifdef _WIN32
    LPOSVERSIONINFO lpVersionInformation = new OSVERSIONINFO;
    lpVersionInformation->dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    if(GetVersionEx(lpVersionInformation))
          return number(lpVersionInformation->dwBuildNumber);
    return string("");
#elif defined (__GNUC__)
    struct utsname buf;
    uname(&buf);
    return string(buf.version);
#else
    #pragma "message OS Version not defined"
    return string("");
#endif
}

string getOSRelease(){
#ifdef _WIN32
        LPOSVERSIONINFO lpVersionInformation = new OSVERSIONINFO;
        lpVersionInformation->dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
        if(GetVersionEx(lpVersionInformation))
                return number(lpVersionInformation->dwMajorVersion)
                                + '.' + number(lpVersionInformation->dwMinorVersion) ;
        return string("");
#elif __GNUC__
    struct utsname buf;
    uname(&buf);
    return string(buf.version);
#else
#pragma "message OS Release not defined"
    return string("");
#endif
}

string getMachineName(){
#ifdef _WIN32
    char lpBuffer[MAX_COMPUTERNAME_LENGTH+1]; // buffer
    DWORD nSize(MAX_COMPUTERNAME_LENGTH+1);
    if(GetComputerNameA(lpBuffer,&nSize))
      return string(lpBuffer);
    return string("");
#elif __GNUC__
    struct utsname buf;
    uname(&buf);
    return string(buf.nodename);
#else
#pragma "message machine name not defined"
    return string("");
#endif
}

string getArchitecture(){
#ifdef _WIN32

#ifdef _M_ALPHA
    return string("DEC ALPHA");
#elif _M_IX86
    int m_i86 = _M_IX86 + 86;
    return string("i") + number(m_i86);
#elif _M_MPPC
    int m_mppc = _M_MPPC;
    return string("Mac Power PC ") + number(m_mppc);
#elif _M_PPC
      int m_ppc = _M_PPC;
      return string("Power PC ") + number(m_ppc);
#elif _M_MRX000
      int m_mrx = _M_MRX000;
      return string("R") + number(m_mrx);
#endif

#elif __GNUC__
    struct utsname buf;
    uname(&buf);
    return string(buf.machine);
#endif

        return string("");
}

string getCompilerName(){
    string c_name;
#if defined (_MSC_VER)
    c_name = "Microsoft Visual C++";
#elif __GNUC__
    c_name = "GNU C++";
#else
#pragma "message compiler name not defined"
#endif
    return c_name;
}

string getCompilerVersion(){
    string c_version;
#if defined (_MSC_VER)
    c_version = number(_MSC_VER) ;
    if(c_version == "1200") c_version = "6.0 (1200)";
#elif defined (__GNUC__)
    c_version = __VERSION__ ;
#else
#pragma message "compiler version not defined"
#endif
        return c_version;
}

string getOSLanguage(){
#ifdef PGL_CORE_WITHOUT_QT
    string lang = "English";
#ifdef _WIN32
      int cchData = GetLocaleInfoA(LOCALE_USER_DEFAULT,LOCALE_SENGLANGUAGE,NULL,0);
      LPSTR lpLCData = new char[cchData];
      cchData = GetLocaleInfoA(LOCALE_USER_DEFAULT,LOCALE_SENGLANGUAGE,lpLCData,cchData);
      lang = string(lpLCData);
#elif __GNUC__
 #pragma message "os language not defined"
#else
 #pragma message "os language not defined"
#endif
    return lang;
#else
    QString locale_lang = QLocale::languageToString(QLocale::system().language());
    return string(qPrintable(locale_lang));
#endif
}

string getLanguage(){
    string lang = getOSLanguage();
    return lang;
}

void setLanguage(const string& lang){
}

PGL_END_NAMESPACE
