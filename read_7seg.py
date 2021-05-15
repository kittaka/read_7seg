#!/usr/bin/env python
# coding: utf-8

# In[103]:


from PIL import Image
import pyocr
import os
import numpy as np
import sys
import pyocr.builders
import cv2
import pytesseract

img_pass="26.0.png"
img_or = cv2.imread(img_pass)

#グレースケール化
img_gray = cv2.cvtColor(img_or, cv2.COLOR_RGB2GRAY)
#2値化（100:２値化の閾値／画像を見て調整する）
ret,thresh1 = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY)
#2値化 ガウシアン何かを間違っていてひどいことに…
#thresh1 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,2)

#2値化（自動で決定）使ってる場合でない精度…
#ret,thresh1 = cv2.threshold(img_gray,100,255,cv2.THRESH_OTSU)

#ノイズ処理（モルフォロジー変換）
kernel = np.ones((2,2),np.uint8)
img_opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
# -*- coding: utf-8 -*-

#cwd = os.getcwd()
#print(cwd)
#img_file="test.png"
#img_pass=cwd+img_file

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

temp_pil_im = cv2pil(img_opening)

#print(pytesseract.image_to_osd(temp_pil_im))
#回転 して補正
temp_pil_im = temp_pil_im.rotate(2,expand=True,fillcolor=255)


#print(temp_pil_im.height,temp_pil_im.width)

def add_margin(pil_img, color):
    width, height = pil_img.size
    new_width = int(width *1.2)
    new_height = int(height *1.2)
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (int(width*0.1), int(height*0.1)))
    return result

#第2引数が0で黒、255で白
temp_pil_im=add_margin(temp_pil_im,255)
#試しに現段階の画像表示    
temp_pil_im.show()
#試しに現段階の画像保存
temp_pil_im.save('tmp.jpg', quality=95)

# OCR エンジン取得
tools = pyocr.get_available_tools()
#print(tools)

tool = tools[0]

#print(tool)



builder=pyocr.builders.TextBuilder(tesseract_layout=8)
#builder.tesseract_configs.append("digits")

# 使用する画像を指定してOCRを実行 テキストを検出
txt1 = tool.image_to_string(
    temp_pil_im,
    lang="eng",
    builder=builder
    #builder=pyocr.tesseract.DigitBuilder(tesseract_layout=3)
    
)

builder=pyocr.tesseract.DigitBuilder(tesseract_layout=8)
#builder.tesseract_configs.append("digits")
# 使用する画像を指定してOCRを実行 数字のみを検出
txt2 = tool.image_to_string(
    temp_pil_im,
    lang="letsgodigital",
    #builder=pyocr.builders.TextBuilder(tesseract_layout=8)
    #builder=pyocr.tesseract.DigitBuilder(tesseract_layout=8)
    builder=builder
    
    
)
# 結果をテキストで出力
with open("result_1.txt", "w") as f:
    print(txt1, file=f)
with open("result_2.txt", "w") as f:
    print(txt2, file=f)  
print(txt1)
print(txt2)


# In[ ]:





# In[ ]:




