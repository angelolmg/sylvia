import cv2
import time
from math import atan, degrees

colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
minscore = 0.5
screen_w = 0
screen_h = 0
screen_area = 0
collision_fov = 90              # Field of "view" of colisions: (max degrees, min degrees)
                                # Every object inside this window could collide to body
collision_distance = 2          # Max distance to alert emminent collision
frontal_collision_fov = 40       # Everything inside this window could collide to the front of the body

# Basic colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (80, 80, 80)
green = (0, 255, 0)
red = (0, 0, 255)

# Direction arrow padding
y_padding = 200
x_padding = 0

# Basic labels
memory_label = f"Memory queue:"
straight_label = f"GO STRAIGHT"
turn_right_label = f"TURN RIGHT"
turn_left_label = f"TURN LEFT"

null_object = ('null', 999, 180)
object_stack = [null_object] * 10
obj_append_delay = 5
obj_append_counter = 0 

class_names = []
with open("coco.names") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("rio_walk2.mp4")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

while True:

    _, frame = cap.read() 

    try:

        if screen_area == 0:
            (screen_w, screen_h, _) = frame.shape
            screen_area = screen_w * screen_h

        start = time.time()
        classes, scores, boxes = model.detect(frame, 0.1, 0.2)
        end = time.time()

        objects = zip(classes, scores, boxes)
        objects = [x for x in objects if x[1] > minscore]

        xpos = 0
        largest_area = 0
        largest_label = ""
        distance = 0
        angle = 0
        largest_box = []


        for (classid, score, box) in objects:
            area = box[2] * box[3]

            # its the largest object
            if area > largest_area:
                largest_area = area
                largest_box = box
                largest_label = class_names[classid[0]]

            color = colors[int(classid) % len(colors)]
            label = f"{class_names[classid[0]]} : {score}"
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

        if len(largest_box) > 0:
            xpos = largest_box[0] + int(largest_box[2]/2)
            ypos = largest_box[1] + int(largest_box[3])
            midy = largest_box[1] + int(largest_box[3]/2)
            #ypos = int(3*screen_w/4)

            # proportion by area
            proportion_area = 2*largest_area/screen_area
            #proportion by ypos
            proportion_ypos = ypos/screen_h

            proportion = min(proportion_area + proportion_ypos, 1)

            close_meter = min(int(255 * proportion), 255)
            tickness = min(max(int(5 * proportion), 2), 5)
            distance = max(round((5 - 5 * proportion), 2), 0)

            # Add one to avoid division by zero
            if xpos < screen_h/2 : angle = round(degrees(atan((screen_h/2 - xpos)/(midy + 1))) + 90, 2)
            else: angle = round(degrees(atan(midy/(xpos - screen_h/2 + 1))), 2)

            cv2.arrowedLine(frame, (int(screen_h/2), screen_w), (xpos, midy), black, tickness+5)
            cv2.arrowedLine(frame, (int(screen_h/2), screen_w), (xpos, midy), (0, 255 - close_meter, close_meter), tickness)


        should_avoid = False
        if  distance < collision_distance and \
            angle < (90 + collision_fov/2) and \
            angle > (90 - collision_fov/2):
            should_avoid = True


        if obj_append_counter >= obj_append_delay:
            if should_avoid: 
                to_append = (largest_label, distance, angle)
            else: to_append = null_object
            object_stack.append(to_append)
            object_stack.pop(0)
            obj_append_counter = 0
        else: obj_append_counter += 1


        frame = cv2.copyMakeBorder(frame, 0, 300, 0, 0, cv2.BORDER_CONSTANT, value=gray)

        focus_obj = f"Focusing now on: {largest_label}"
        focus_distance = f"Aproximate distance: {distance} meters"
        focus_angle = f"Aproximate angle: {angle} degrees"
        avoid_flag = f"Should avoid? {should_avoid}"
        fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
        

        if largest_label == "":
            focus_obj = f"Focusing now on: Nothing"
            focus_distance = f"Aproximate distance: None"
            focus_angle = f"Aproximate angle: None"
            avoid_flag = f"Should avoid? {False}"
        
        cv2.putText(frame, fps_label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, black, 4)
        cv2.putText(frame, fps_label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, green, 2)
        cv2.putText(frame, focus_obj, (10, screen_w + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        cv2.putText(frame, focus_distance, (10, screen_w + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        cv2.putText(frame, focus_angle, (10, screen_w + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        cv2.putText(frame, avoid_flag, (10, screen_w + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        cv2.putText(frame, memory_label, (10, screen_w + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        

        # Initializing direction arow colors
        lcolor = green
        rcolor = green
        ucolor = green


        # Display memory queue
        # Updating arrow colors depending on possible collisions
        for i in range(len(object_stack)):
            obj_label = f"{i+1} : {object_stack[i][0]} -> {object_stack[i][1]} m | {object_stack[i][2]} dgrs"
            cv2.putText(frame, obj_label, (10, screen_w + 150 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)
            if object_stack[i][0] != "null":
                if   object_stack[i][2] < (90 - frontal_collision_fov/2):   rcolor = red
                elif object_stack[i][2] > (90 + frontal_collision_fov/2):   lcolor = red
                else:                                                       ucolor = red


        # Display left arrow
        cv2.arrowedLine(frame, (int(5*screen_h/8) - x_padding, screen_w + y_padding), (int(screen_h/2) - x_padding, screen_w + y_padding), white, 25, tipLength=0.5)
        cv2.arrowedLine(frame, (int(5*screen_h/8) - x_padding, screen_w + y_padding), (int(screen_h/2) - x_padding, screen_w + y_padding), lcolor, 15, tipLength=0.5)
        cv2.putText(frame, turn_left_label, (int(screen_h/2) - x_padding, screen_w + y_padding + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

        # Display right arrow
        cv2.arrowedLine(frame, (int(3*screen_h/4) - x_padding, screen_w + y_padding), (int(7*screen_h/8) - x_padding, screen_w + y_padding), white, 25, tipLength=0.5)
        cv2.arrowedLine(frame, (int(3*screen_h/4) - x_padding, screen_w + y_padding), (int(7*screen_h/8) - x_padding, screen_w + y_padding), rcolor, 15, tipLength=0.5)
        cv2.putText(frame, turn_right_label, (int(3*screen_h/4) - x_padding, screen_w + y_padding + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

        # Display up arrow
        cv2.arrowedLine(frame, (int(.6875*screen_h) - x_padding, screen_w + y_padding - 30), (int(.6875*screen_h) - x_padding, int(.75*screen_w + y_padding - 30)), white, 25, tipLength=0.5)
        cv2.arrowedLine(frame, (int(.6875*screen_h) - x_padding, screen_w + y_padding - 30), (int(.6875*screen_h) - x_padding, int(.75*screen_w + y_padding - 30)), ucolor, 15, tipLength=0.5)
        cv2.putText(frame, straight_label, (int(.6875*screen_h) - x_padding - 50, int(.75*screen_w + y_padding - 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
        

        cv2.imshow("SYLVIA", frame)
    
    except:
        print("Null frame")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()