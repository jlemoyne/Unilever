
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='jclaudel', api_key='nRCYY30Z8RkJfoHp9n43')

# plotly.tools.set_config_file(world_readable=True, sharing='public')

demo_csv = '/Users/jeanclaudelemoyne/work/Data/Unilever/presentation/apg_demo_volume.csv'
m44_demo_plot = '/Users/jeanclaudelemoyne/work/Data/Unilever/presentation/m44volume'


def eg1():

    trace0 = go.Scatter(
        x=[1, 2, 3, 4],
        y=[10, 15, 13, 17]
    )
    trace1 = go.Scatter(
        x=[1, 2, 3, 4],
        y=[16, 5, 11, 9]
    )
    data = [trace0, trace1]

    py.plot(data, filename = 'basic-line', auto_open=True)


def plot_demo(demo_csv, demo_plot):
    df = pd.read_csv(demo_csv)
    # print '~~~~~~> ', type(x), type(y), len(x), len(y)
    trace0 = go.Scatter(
        y = df['volume'].tolist(),
        x = df['M2017.35.44'].tolist()
    )

    data = [trace0]
    layout = go.Layout(title='Ice-cream Volume by Male population age group 35 to 44',
                       xaxis=dict(
                           title='Population group Male 35 to 44',
                           titlefont=dict(
                               family='Arial, Courier New, monospace',
                               size=18,
                               color='#7f7f7f'
                           )
                        ),
                       yaxis=dict(
                           title='Ice-cream Volume',
                           titlefont=dict(
                               family='Arial, Courier New, monospace',
                               size=18,
                               color='#7f7f7f'
                           )
                       )
                    )
    fig = go.Figure(data=data,layout=layout)
    # py.image.save_as({'data':data}, demo_plot, format='png')
    py.image.save_as(fig, demo_plot, format='png')
    # py.iplot(data, filename=demo_plot)


if __name__ == '__main__':
    print 'Unilever presentation material ...'
    # eg1()
    plot_demo(demo_csv, m44_demo_plot)