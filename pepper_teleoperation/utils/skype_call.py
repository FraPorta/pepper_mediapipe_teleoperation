from concurrent.futures import thread
import time
import os
import pyautogui
import webbrowser
from threading import Thread

def make_skype_call():
    # Open Chrome browser
    filename = "GUI_material\skype_call.html"
    # chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # chrome_open = webbrowser.get(chrome_path).open_new_tab(os.path.realpath(filename))
    # if not chrome_open:
    webbrowser.open(os.path.realpath(filename))

    # Wait for the browser to open
    time.sleep(1.5)
    
    # Click automatically the 'Open Skype' button 
    pyautogui.press('left')
    pyautogui.press('enter')
    
    with pyautogui.hold('ctrl'):
        pyautogui.press('w')
            

def run_skype_thread():
    thread = Thread(target=make_skype_call)
    thread.start()
    thread.join()
    print("Skype call made")

if __name__ == '__main__':
    run_skype_thread()
