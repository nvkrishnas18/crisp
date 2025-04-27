import plotly.express as px
import json

def plotly_to_echarts(fig):
    echarts_json = {
        "title": {"text": fig.layout.title.text if fig.layout.title else ""},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": []},
        "xAxis": {"type": "category", "data": []},
        "yAxis": {"type": "value"},
        "series": []
    }
    
    for trace in fig.data:
        series_data = {
            "name": trace.name if trace.name else "Series",
            "type": "line" if trace.mode == "lines" else "bar",
            "data": []
        }
        
        if "x" in trace:
            echarts_json["xAxis"]["data"] = trace.x.tolist()
        print("type trace.y",type(trace.y))
        if "y" in trace:
            series_data["data"] = trace.y.tolist()
        
        echarts_json["series"].append(series_data)
        echarts_json["legend"]["data"].append(series_data["name"])
    print("echarts_json::",type(echarts_json))
    return echarts_json

# Example usage:
fig = px.line(x=[1, 2, 3, 4], y=[10, 20, 15, 25], title="Example Plotly Chart")
echarts_json = plotly_to_echarts(fig)
print(echarts_json)