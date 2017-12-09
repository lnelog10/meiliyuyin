import os;
import glob as gb

count = 1
img_path = gb.glob("*.mp4")
for file in img_path:
    if os.path.isdir(file):  # 如果是文件夹则跳过
         continue;
    name= "%04g" % count
    print("rename "+file +"to:"+name)
    os.rename(file, name+".mp4");  # 重命名
    count += 1;

img_path = gb.glob("*.rmvb")
for file in img_path:
    if os.path.isdir(file):  # 如果是文件夹则跳过
         continue;
    name= "%04g" % count
    print("rename "+file +" to:"+name+".mp4")
    strFileName = "'" + file.replace("'", "'\\''") + "'"
    # print(strFileName)
    os.system("ffmpeg -i "+strFileName+" "+name+".mp4")
    os.remove(file)
    count += 1;

img_path = gb.glob("*.flv")
for file in img_path:
    if os.path.isdir(file):  # 如果是文件夹则跳过
         continue;
    name= "%04g" % count
    print("rename "+file +" to:"+name+".mp4")
    strFileName = "'" + file.replace("'", "'\\''") + "'"
    # print(strFileName)
    os.system("ffmpeg -i "+strFileName+" "+name+".mp4")
    os.remove(file)
    count += 1;