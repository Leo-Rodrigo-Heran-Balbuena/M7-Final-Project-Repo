import cv2


cam = cv2.VideoCapture("videos/counterspell.mp4")

frameno = 0
while(True):
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = 'counterspell_' + str(frameno) + '.jpg'
      print ('new frame captured...' + name)
      name = 'from_video/' + name
      cv2.imwrite(name, frame)
      frameno += 1
   else:
      break

cam.release()
cv2.destroyAllWindows()

