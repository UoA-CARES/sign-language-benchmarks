class ReadPose:
    def __init__(self):
        self.keypoints = ["nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
                ]

    def __call__(self, line):
        line = [l for l in line.replace(',',"").split(' ') if l != '' and l != '\n']
        imgpath = line[0]
        line = line[1:]
        line = [float(l) for l in line]


        #line [imgpath [0], [y,x, conf][1], ...[4], ... [52], headlb[54], headrt,lhandlb[58], lhandrt, rhandlb[62], rhandrt, bboxlb[66], bboxrt[68] ]

        if len(line) != 0:
            posepoints = line[0:51]
            head = line[51:55]
            lhand =line[55:59]
            rhand = line[59:63]
            bodybbox = line[63:]
        else:
            head = [0., 0., 255., 255.]
            lhand = [0., 0., 255., 255.]
            rhand = [0., 0., 255., 255.]
            bodybbox = [0., 0., 255., 255.]
            posepoints = line[0:51]
        
            

        pose_values = dict()
        for i in range(0, 51, 3):
            if len(posepoints) == 0:
                pose_values[self.keypoints[i//3]] = dict(y = 0.,
                                            x=0.,
                                            confidence=0.)
            else:
                pose_values[self.keypoints[i//3]] = dict(y = posepoints[i],
                                                x=posepoints[i+1],
                                                confidence=posepoints[i+2])
        
        return pose_values, head, lhand, rhand, bodybbox, imgpath