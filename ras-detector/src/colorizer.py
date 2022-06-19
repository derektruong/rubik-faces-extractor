import numpy as np
import cv2
from color_detector import color_detector
from constants import STICKER_CONTOUR_COLOR, STICKER_AREA_TILE_SIZE, STICKER_AREA_TILE_GAP, STICKER_AREA_OFFSET, COLOR_PALETTE

class Colorizer:
    
    def __init__(self):
        self.average_sticker_colors = {}
        self.preview_state  = [(255,255,255), (255,255,255), (255,255,255),
                                (255,255,255), (255,255,255), (255,255,255),
                                (255,255,255), (255,255,255), (255,255,255)]
    
    def find_contours(self, dilatedFrame):
        """Find the contours of a 3x3x3 cube."""
        contours, hierarchy = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []

        # Step 1/4: filter all contours to only those that are square-ish shapes.
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
            
            if len (approx) == 4:
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)

                # Find aspect ratio of boundary rectangle around the countours.
                ratio = w / float(h)

                # Check if contour is close to a square.
                if ratio >= 0.6 and ratio <= 1.4 and w >= 25 and w <= 70 and area / (w * h) > 0.5:
                    final_contours.append((x, y, w, h))

        # Return early if we didn't found 9 or more contours.
        if len(final_contours) < 9:
            return []

        # Step 2/4: Find the contour that has 9 neighbors (including itself)
        # and return all of those neighbors.
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2
            center_y = y + h / 2
            radius = 1.5

            # Create 9 positions for the current contour which are the
            # neighbors. We'll use this to check how many neighbors each contour
            # has. The only way all of these can match is if the current contour
            # is the center of the cube. If we found the center, we also know
            # all the neighbors, thus knowing all the contours and thus knowing
            # this shape can be considered a 3x3x3 cube. When we've found those
            # contours, we sort them and return them.
            neighbor_positions = [
                # top left
                [(center_x - w * radius), (center_y - h * radius)],

                # top middle
                [center_x, (center_y - h * radius)],

                # top right
                [(center_x + w * radius), (center_y - h * radius)],

                # middle left
                [(center_x - w * radius), center_y],

                # center
                [center_x, center_y],

                # middle right
                [(center_x + w * radius), center_y],

                # bottom left
                [(center_x - w * radius), (center_y + h * radius)],

                # bottom middle
                [center_x, (center_y + h * radius)],

                # bottom right
                [(center_x + w * radius), (center_y + h * radius)],
            ]

            for neighbor in final_contours:
                (x2, y2, w2, h2) = neighbor
                cv2.rectangle(
                    dilatedFrame,
                    (x2, y2),
                    (x2 + w2, y2 + h2),
                    color=(0, 255, 0),
                    thickness = 2
                )
                for (x3, y3) in neighbor_positions:
                    # The neighbor_positions are located in the center of each
                    # contour instead of top-left corner.
                    # logic: (top left < center pos) and (bottom right > center pos)
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        contour_neighbors[index].append(neighbor)

        # Step 3/4: Now that we know how many neighbors all contours have, we'll
        # loop over them and find the contour that has 9 neighbors, which
        # includes it This is the center piece of the cube. If we come
        # across it, then the 'neighbors' are actually all the contours we're
        # looking for.
        for (contour, neighbors) in contour_neighbors.items():
            if len(neighbors) == 9:
                found = True
                final_contours = neighbors
                break

        if not found:
            return []

        # Step 4/4: When we reached this part of the code we found a cube-like
        # contour. The code below will sort all the contours on their X and Y
        # values from the top-left to the bottom-right.

        # Sort contours on the y-value first.
        y_sorted = sorted(final_contours, key=lambda item: item[1])

        # Split into 3 rows and sort each row on the x-value.
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    def draw_contours(self, frame, contours):
        """Draw contours onto the given frame."""
        for _, (x, y, w, h) in enumerate(contours):
            cv2.rectangle(frame, (x, y), (x + w, y + h), STICKER_CONTOUR_COLOR, 2)
            
    def draw_preview_stickers(self, frame):
        """Draw the current preview state onto the given frame."""
        self.draw_stickers(frame, self.preview_state, STICKER_AREA_OFFSET, STICKER_AREA_OFFSET)
            
    def draw_stickers(self, frame, stickers, offset_x, offset_y):
        """Draws the given stickers onto the given frame."""
        index = -1
        for row in range(3):
            for col in range(3):
                index += 1
                x1 = (offset_x + STICKER_AREA_TILE_SIZE * col) + STICKER_AREA_TILE_GAP * col
                y1 = (offset_y + STICKER_AREA_TILE_SIZE * row) + STICKER_AREA_TILE_GAP * row
                x2 = x1 + STICKER_AREA_TILE_SIZE
                y2 = y1 + STICKER_AREA_TILE_SIZE

                # shadow
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1
                )
                # foreground color
                cv2.rectangle(
                    frame,
                    (x1 + 1, y1 + 1),
                    (x2 - 1, y2 - 1),
                    color_detector.get_prominent_color(stickers[index]),
                    -1
                )
                
    def update_preview_state(self, frame, contours):
        """
        Get the average color value for the contour for every X amount of frames
        to prevent flickering and more precise results.
        """
        max_average_rounds = 8
        for index, (x, y, w, h) in enumerate(contours):
            if index in self.average_sticker_colors and len(self.average_sticker_colors[index]) == max_average_rounds:
                sorted_items = {}
                for bgr in self.average_sticker_colors[index]:
                    key = str(bgr)
                    if key in sorted_items:
                        sorted_items[key] += 1
                    else:
                        sorted_items[key] = 1
                most_common_color = max(sorted_items, key=lambda i: sorted_items[i])
                self.average_sticker_colors[index] = []
                self.preview_state[index] = eval(most_common_color)
                break

            roi = frame[y+7:y+h-7, x+14:x+w-14]
            avg_bgr = color_detector.get_dominant_color(roi)
            closest_color = color_detector.get_closest_color(avg_bgr)['color_bgr']
            self.preview_state[index] = closest_color
            if index in self.average_sticker_colors:
                self.average_sticker_colors[index].append(closest_color)
            else:
                self.average_sticker_colors[index] = [closest_color]
    
    def automatic_brightness_and_contrast(self, image, clip_hist_percent=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result, alpha, beta)
    
    def mapping_color_name(self):
        results = []
        for color in self.preview_state:
            for k, v in COLOR_PALETTE.items():
                if color == v:
                    results.append(k)
                    break
                
        return results
    
    def colorize(self, path, current_face):
        frame = cv2.imread(path)
        frame = cv2.resize(frame,(250,250),interpolation=cv2.INTER_BITS)
        
        frame, alpha, beta = self.automatic_brightness_and_contrast(frame)
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurredFrame = cv2.blur(grayFrame, (3, 3))
        cannyFrame = cv2.Canny(blurredFrame, 30, 60, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilatedFrame = cv2.dilate(cannyFrame, kernel)

        contours = self.find_contours(dilatedFrame)
        is_success = False
        if len(contours) == 9:
            self.draw_contours(frame, contours)
            self.update_preview_state(frame, contours)
            self.draw_preview_stickers(frame)
            
            print(f"{current_face}: {self.mapping_color_name()}")
            cv2.imshow(f"{current_face}",frame)
            cv2.moveWindow(f"{current_face}", 1000,300)
            is_success = True
        
        if not is_success:
            print("Not detected, please try again!")
            return False
        return True
        
colorizer = Colorizer()
