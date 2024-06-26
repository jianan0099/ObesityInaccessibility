import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None
pio.renderers.default = 'browser'

# ---- font settings ------
font = {'family': 'Helvetica',
        'color': 'black',
        }

# -------------  map settings ----------------------
# Read US states shape file
statesmap = gpd.read_file('shapefile/tl_2017_us_state.shp')
statesmap = statesmap.to_crs(epsg=2163)

# Select rows to adjust geometries.
al = statesmap.NAME == 'Alaska'
hi = statesmap.NAME == 'Hawaii'
pr = statesmap.NAME == 'Puerto Rico'
# adjust geometries
statesmap[al] = statesmap[al].set_geometry(statesmap[al].
                                           scale(.2, .2, .2).translate(1.4e6, -4.5e6))
statesmap[hi] = statesmap[hi].set_geometry(statesmap[hi].
                                           scale(.8, .8, .8).translate(5e6, -1.35e6))
statesmap[pr] = statesmap[pr].set_geometry(statesmap[pr].
                                           scale(2, 2, 2).translate(-2.3e6, 0.3e6))

loc_text = []
for loc in statesmap['STUSPS']:
    if loc in ['MD', 'DE', 'NJ', 'CT', 'RI', 'NH']:
        loc_text.append('')
    else:
        loc_text.append(loc)

# -------------- fig 3 ------------------------

# read state weights
death_weight_df = pd.read_excel('data/fig3_death_weight.xlsx', sheet_name='detailed')
death_per = 100000
death_weight = {}
for _, row in death_weight_df.iterrows():
    death_weight[row['STUSPS']] = {'eligible_pop_frac (adjust)': row['eligible_pop_frac (adjust)']*100,
                                   'total_death_per100000': row['total_death_per100000']}

# add data to statesmap
for index, row in statesmap.iterrows():
    if row['STUSPS'] in death_weight:
        statesmap.loc[index, 'eligible_pop_frac (adjust)'] = death_weight[row['STUSPS']]['eligible_pop_frac (adjust)']
        statesmap.loc[index, 'total_death_per100000'] = death_weight[row['STUSPS']]['total_death_per100000']
    else:
        for keys in ['eligible_pop_frac (adjust)', 'total_death_per100000',]:
            statesmap.loc[index, keys] = None
death_per_capita = statesmap['total_death_per100000']
cb_min, cb_max = death_per_capita.min(), death_per_capita.max()

#  draw
fig = px.choropleth(statesmap,
                    locations='STUSPS',
                    color='total_death_per100000',
                    color_continuous_scale='spectral_r',
                    locationmode='USA-states',
                    scope='usa'
                    )
fig.add_scattergeo(
    locations=statesmap['STUSPS'],
    locationmode='USA-states',
    text=loc_text,
    textfont=font,
    mode='text')

fig.update_layout(  # coloraxis_colorbar_x=0.9, font=font,
    margin=dict(l=0, r=0, t=0, b=0, pad=0),
    coloraxis=dict(cmax=cb_max, cmin=cb_min))
fig.update_geos(showlakes=False, visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig.write_image("figs/fig3_map.pdf")

# --------------- fig 1 map -----------------
loc_text = []
for loc in statesmap['STUSPS']:
    if loc in ['MD', 'DE', 'NJ', 'CT', 'RI', 'NH']:
        loc_text.append('')
    else:
        loc_text.append(loc)
fig = px.choropleth(statesmap,
                    locations='STUSPS',
                    color='eligible_pop_frac (adjust)',
                    color_continuous_scale='spectral_r',
                    locationmode='USA-states',
                    scope='usa'
                    )
c = fig.add_scattergeo(
    locations=statesmap['STUSPS'],
    locationmode='USA-states',
    text=loc_text,
    textfont=font,
    mode='text')
font = {'family': 'Helvetica',
        'color': 'black',
        'size': 17,
        }
fig.update_layout(coloraxis_colorbar_title_text='',
                  coloraxis_colorbar_title_side='bottom',
                  font=font, margin=dict(l=0, r=0, t=0, b=0),
                  geo=dict(showlakes=False, visible=False))
fig.update_coloraxes(
    colorbar={'orientation': 'h', 'x': 0.5, 'y': -0.1, 'len': 0.8, 'ticks': 'outside', 'thickness': 20})
fig.show()
fig.write_image("figs/fig1_map.pdf")
