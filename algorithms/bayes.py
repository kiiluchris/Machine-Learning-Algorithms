# numbers representing low, middle, and high income

all_cars = [
    [25, 30, 40],
    [15, 65, 75]
]


col_totals = [sum(col) for col in zip(*all_cars)]
row_totals = [sum(row) for row in all_cars]
all_totals = sum(row_totals)

row_prob = [[
    car_ / total for car_ in cars
    ] for cars, total in zip(all_cars, row_totals)]

col_prob = [[
    car_ / total for car_ in cars
    ] for cars, total in zip(zip(*all_cars), col_totals)]
    
marginal_prob =  [[
    car_ / all_totals for car_ in cars
    ] for cars in all_cars]



print(marginal_prob)