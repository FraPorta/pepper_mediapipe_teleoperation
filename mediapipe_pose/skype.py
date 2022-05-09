import sys
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import pyautogui

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # base_path = os.path.abspath(".")
        base_path = os.getcwd()
        
    return os.path.join(base_path, relative_path)

browser_selection = [
                        lambda : webdriver.Chrome(),
                        lambda : webdriver.Edge(),
                        lambda : webdriver.ChromiumEdge(),
                        lambda : webdriver.Firefox(),
                        lambda : webdriver.Safari(),
                    ]

driver = None            
for browser in browser_selection:
    try:
        driver = browser()
    except selenium.common.exceptions.WebDriverException as e:
        print("Couldn't open the Skype link with any browser")
    if driver:
        break
  
filename = resource_path("skype_call.html")
driver.get("file://" + os.path.realpath(filename))
button = driver.find_element(by=By.LINK_TEXT, value="Call")
button.click() 

time.sleep(2)

# Click automatically the 'Open Skype' button 
start = pyautogui.locateCenterOnScreen('open_skype.png')#If the file is not a png file it will not work 
if start:
    pyautogui.click(start) #Moves the mouse to the coordinates of the image

# Wait 5 seconds and then exit from the browser
time.sleep(5)
driver.close()