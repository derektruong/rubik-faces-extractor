import twophase.solver  as sv

class CubeSolver:
    def transformationColorFaces(self, up, right, front, down, left, back):
        
        left_1 = [left[6], left[3], left[0], left[7], left[4], left[1], left[8], left[5], left[2]]
        left = left_1

        right_1 = [right[2], right[5], right[8], right[1], right[4], right[7], right[0], right[3], right[6]]
        right = right_1

        back_1 = [back[8], back[7], back[6], back[5], back[4], back[3], back[2], back[1], back[0]]
        back = back_1

        faces = [up, right, front, down, left, back]

        cubestring = ''

        for face in faces:
            for facelet in face:
                if facelet == 'yellow':
                    cubestring = cubestring + 'U'
                    continue
                if facelet == 'blue':
                    cubestring = cubestring + 'R'
                    continue
                if facelet == 'orange':
                    cubestring = cubestring + 'F'
                    continue
                if facelet == 'white':
                    cubestring = cubestring + 'D'
                    continue
                if facelet == 'green':
                    cubestring = cubestring + 'L'
                    continue
                if facelet == 'red':
                    cubestring = cubestring + 'B'
                    continue
                
        return sv.solve(cubestring,19,2)
    

solver = CubeSolver()