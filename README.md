# ESGF Melbourne (dev)

## Data

Minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia.

![Daily temperature evolution](docs/pics/temperature_evolution.png "Daily Temperature Evolution")

## Objective

We want to build a model able to predict the **daily** temperature in Melbourne over the **next year**.

## Evaluation

Fit one or more models using the data strictly anterior to the evaluation years 
`t=1987`, `t=1988` and `t=1989` in order to predict the daily temperature 
during the year `t=1987`, `t=1988` and `t=1989` respectively.

### Results

Choose a suitable metric to evaluate the performance of your models. 
Report the metric in the following table. 

For each evaluation year `t`, you must report the performance considering `k=3`, `k=6` and `k=12` months of data following `t`.

<table>
    <thead>
        <tr>
            <th>Evaluation year</th>
            <th>Next 3 months</th>
            <th>Next 6 months</th>
            <th>Next 12 months</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1987</td>
            <td>loss: 12.6189 - mean_squared_error: 5.8430</td>
            <td>loss: 21.4359 - mean_squared_error: 11.4522</td>
            <td>loss: 25.5057 - mean_squared_error: 12.8415</td>
        </tr>
        <tr>
            <td>1988</td>
            <td>loss: 15.1681 - mean_squared_error: 8.3459</td>
            <td>loss: 22.8974 - mean_squared_error: 11.4532</td>
            <td>loss: 29.6565 - mean_squared_error: 16.9872</td>
        </tr>
        <tr>
            <td>1989</td>
            <td>loss: 20.2148 - mean_squared_error: 13.3047</td>
            <td>loss: 23.7902 - mean_squared_error: 12.6433</td>
            <td>loss: 31.6298 - mean_squared_error: 18.9534</td>
        </tr>
    </tbody>
</table>
