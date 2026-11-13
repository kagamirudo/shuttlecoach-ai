"""Store only this app's Groq credential in macOS Keychain, never a project file."""
import ctypes as C
import os
import sys
from pathlib import Path

SERVICE='ShuttleCoach Groq API'
ACCOUNT='groq-api-key'


def clean_key(value):
    key=value.strip()
    if not key:raise ValueError('Enter a Groq API key first.')
    if len(key)>512 or any(not 33<=ord(c)<=126 for c in key):raise ValueError('The API key must be a single printable token without whitespace.')
    return key


def _keychain(action, value=None):
    if sys.platform!='darwin':
        if action=='load':return None
        raise ValueError('Remembering API keys requires macOS Keychain. Use GROQ_API_KEY on other platforms.')
    cf=C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    security=C.CDLL('/System/Library/Frameworks/Security.framework/Security')
    pointer=C.c_void_p;owned=[]
    cf.CFStringCreateWithCString.argtypes=[pointer,C.c_char_p,C.c_uint32];cf.CFStringCreateWithCString.restype=pointer
    cf.CFDictionaryCreate.argtypes=[pointer,C.POINTER(pointer),C.POINTER(pointer),C.c_long,pointer,pointer];cf.CFDictionaryCreate.restype=pointer
    cf.CFDataCreate.argtypes=[pointer,C.c_char_p,C.c_long];cf.CFDataCreate.restype=pointer
    cf.CFDataGetLength.argtypes=[pointer];cf.CFDataGetLength.restype=C.c_long
    cf.CFDataGetBytePtr.argtypes=[pointer];cf.CFDataGetBytePtr.restype=pointer
    cf.CFRelease.argtypes=[pointer]
    for name,args in [('SecItemCopyMatching',[pointer,C.POINTER(pointer)]),('SecItemAdd',[pointer,C.POINTER(pointer)]),('SecItemUpdate',[pointer,pointer]),('SecItemDelete',[pointer])]:
        method=getattr(security,name);method.argtypes=args;method.restype=C.c_int32
    def constant(name):return pointer.in_dll(security,name).value
    def string(text):
        result=cf.CFStringCreateWithCString(None,text.encode(),0x08000100);owned.append(result);return result
    def dictionary(items):
        keys=(pointer*len(items))(*items.keys());values=(pointer*len(items))(*items.values())
        result=cf.CFDictionaryCreate(None,keys,values,len(items),None,None);owned.append(result);return result
    query={constant('kSecClass'):constant('kSecClassGenericPassword'),constant('kSecAttrService'):string(SERVICE),constant('kSecAttrAccount'):string(ACCOUNT)}
    try:
        if action=='load':
            query.update({constant('kSecReturnData'):pointer.in_dll(cf,'kCFBooleanTrue').value,
                          constant('kSecMatchLimit'):constant('kSecMatchLimitOne'),
                          constant('kSecUseAuthenticationUI'):constant('kSecUseAuthenticationUIFail')})
            result=pointer();status=security.SecItemCopyMatching(dictionary(query),C.byref(result))
            if status==-25300:return None
            if status==0:
                try:return C.string_at(cf.CFDataGetBytePtr(result),cf.CFDataGetLength(result)).decode()
                finally:cf.CFRelease(result)
        elif action=='save':
            encoded=clean_key(value).encode();data=cf.CFDataCreate(None,encoded,len(encoded));owned.append(data)
            update={constant('kSecValueData'):data}
            status=security.SecItemUpdate(dictionary(query),dictionary(update))
            if status==-25300:status=security.SecItemAdd(dictionary(query|update),None)
        elif action=='delete':
            status=security.SecItemDelete(dictionary(query))
            if status==-25300:return None
        else:raise ValueError('Unknown credential operation.')
        if status!=0:raise ValueError(f'macOS Keychain could not {action} the API key (status {status}). The key was not written to a file.')
    finally:
        for item in reversed(owned):cf.CFRelease(item)


def can_save_key():return sys.platform=='darwin'


def load_key():
    secret=os.environ.get('GROQ_API_KEY_FILE')
    if secret:
        try:return clean_key(Path(secret).read_text())
        except OSError:raise ValueError('The configured Groq runtime secret could not be read.') from None
    return _keychain('load')
def save_key(value):return _keychain('save',clean_key(value))
def delete_key():return _keychain('delete')


def resolve_key(value=None):
    key=(value or '').strip() or os.environ.get('GROQ_API_KEY','').strip() or load_key()
    if not key:raise ValueError('Add your Groq API key in the app, save it to Keychain, or set GROQ_API_KEY. Never paste it into chat.')
    return clean_key(key)
