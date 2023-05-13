import pandas as pd
import math
import random

def distance(point1: tuple[float, float], point2: tuple[float, float]):
    diff_x = point1[0]-point2[0]
    diff_y = point1[1]-point2[1]

    return math.sqrt(diff_x**2 + diff_y**2)

class Circle:
    def __init__(self, center, points) -> None:
        self.center = center
        self.points = points

    def recalculate_center(self, points):
        coord_points = []
        for point_ind in self.points:
            coord_points.append(points[point_ind][1])
        
        x_sum = 0.0
        y_sum = 0.0

        for point in coord_points:
            x_sum += point[0]
            y_sum += point[1]
        try:
            self.center = (x_sum/len(coord_points), y_sum/len(coord_points))
        except ZeroDivisionError:
            self.center = (0.0, 0.0)

ml_dataset = pd.read_csv("tweets_coord_separated.csv")

latitudes = list(ml_dataset["tweet_lat"])
longitudes = list(ml_dataset["tweet_long"])

points = zip(latitudes, longitudes)

to_drop_rows = []
# filtrar
for (i, row) in ml_dataset.iterrows():
    row = float(row["tweet_lat"]), float(row["tweet_long"])
    # filtrar basura (no float)
    if (not isinstance(row[0], float)) or (not isinstance(row[1], float)):
        to_drop_rows.append(i)
        continue

    # eliminar nans
    if math.isnan(row[0]) or math.isnan(row[1]):
        to_drop_rows.append(i)
        continue

    # a veces en vez de nan, se ponen las coordenadas [0.0, 0.0]
    if row[0] == 0.0 and row[1] == 0.0:
        to_drop_rows.append(i)
        continue

circles_column = []
sizes_column = []
for _ in range(0, len(ml_dataset)):
    circles_column.append(None)
    sizes_column.append(None)

print(len(ml_dataset))
print(len(to_drop_rows))
ml_dataset = ml_dataset.drop(index=to_drop_rows)
print(len(ml_dataset))

filtered_points = []
for (i, row) in ml_dataset.iterrows():
    row = float(row["tweet_lat"]), float(row["tweet_long"])
    filtered_points.append((i, row))

point_amount = len(ml_dataset)

def get_circle_points_count(circles: list[Circle]):
    total = 0
    for circle in circles:
        total += len(circle.points)
    
    return total

def get_unused_points(circles: list[Circle], point_amount: int):
    used_points = []
    for circle in circles:
        used_points.extend(circle.points)

    unused_points = []
    used_points.sort()
    last_point = -1
    for point in used_points:
        if point != last_point + 1:
            for i in range(last_point+1, point):
                unused_points.append(i)
        last_point = point

    for i in range(last_point+1, point_amount):
        unused_points.append(i)
    
    return unused_points

circles: list[Circle] = []

minimum_distance = 2.5
print(point_amount)
print(get_circle_points_count(circles))
while get_circle_points_count(circles) < point_amount:
    print(f"Hay {len(circles)} círculos ya")
    # crear un nuevo círculo
    unused_points = get_unused_points(circles, point_amount)
    selected = random.choice(unused_points)
    circles.append(Circle(filtered_points[selected][1], {}))

    # limpiar los círculos
    for circle in circles:
        circle.points = set()

    # meter los puntos en los círculos
    for (point_ind, (_, point)) in enumerate(filtered_points):
        distances = []
        for (i, circle) in enumerate(circles):
            point_distance = distance(circle.center, point)
            if point_distance < minimum_distance:
                distances.append((i, point_distance))
        
        distances.sort(key= lambda x: x[1])
        if len(distances) > 0:
            circles[distances[0][0]].points.add(point_ind)

    # recalcular centros
    for circle in circles:
        circle.recalculate_center(filtered_points)

for (i, circle) in enumerate(circles):
    size = len(circle.points) ** 2
    for (point) in circle.points:
        pd_index = filtered_points[point][0]
        circles_column[pd_index] = i
        sizes_column[pd_index] = size

for i in reversed(to_drop_rows):
    del circles_column[i]
    del sizes_column[i]

ml_dataset["circle"] = circles_column
ml_dataset["tweet_amount"] = sizes_column

ml_dataset.to_csv("final_circles.csv")
