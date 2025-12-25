
import sys
import os
import json
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_multi_index import analyze_overpricing, INDICES_TO_TRACK

def create_index_plot():
    print("Gathering Index Analysis Data...")
    
    chart_data = []
    json_file = "index_analysis.json"
    
    if os.path.exists(json_file):
        print(f"Loading data from {json_file}...")
        with open(json_file, "r") as f:
            raw_data = json.load(f)
            
        for res in raw_data:
            chart_data.append({
                "Index": res["Index"],
                "Mispricing": res["Mispricing"],
                "MispricingPct": f"{res['Mispricing']*100:.2f}%",
                "Color": "green" if res["Mispricing"] > 0 else "red",
                "Status": res["Status"],
                "TotalActual": f"${res['TotalActual']/1e9:,.1f}B",
                "TotalPredicted": f"${res['TotalPredicted']/1e9:,.1f}B",
                "Count": res["Count"],
                "OfficialCount": res.get("OfficialCount", 0)
            })
    else:
        print("⚠ index_analysis.json not found. Generating fresh data (counts may be zero).")
        for idx in INDICES_TO_TRACK:
            res = analyze_overpricing(idx) # official_count defaults to 0
            if res:
                chart_data.append({
                    "Index": res["Index"],
                    "Mispricing": res["Mispricing"],
                    "MispricingPct": f"{res['Mispricing']*100:.2f}%",
                    "Color": "green" if res["Mispricing"] > 0 else "red",
                    "Status": res["Status"],
                    "TotalActual": f"${res['TotalActual']/1e9:,.1f}B",
                    "TotalPredicted": f"${res['TotalPredicted']/1e9:,.1f}B",
                    "Count": res["Count"],
                    "OfficialCount": 0
                })
            
    if not chart_data:
        print("No index data found to plot.")
        return

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Index Mispricing Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        #plot {{ width: 100%; height: 600px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Global Index Mispricing Analysis</h2>
        <p>Market Cap Weighted Aggregates</p>
    </div>
    
    <div id="plot"></div>
    
    <script>
        var data = {json.dumps(chart_data)};
        
        var trace = {{
            x: data.map(d => d.Index),
            y: data.map(d => d.Mispricing),
            type: 'bar',
            marker: {{
                color: data.map(d => d.Color)
            }},
            text: data.map(d => d.MispricingPct),
            textposition: 'auto',
            hovertemplate: 
                "<b>%{{x}}</b><br>" + 
                "Mispricing: %{{text}}<br>" + 
                "Status: %{{customdata[0]}}<br>" + 
                "Coverage: %{{customdata[3]}} / %{{customdata[4]}}<br>" +
                "Actual: %{{customdata[1]}}<br>" + 
                "Predicted: %{{customdata[2]}}<br>" + 
                "<extra></extra>",
            customdata: data.map(d => [d.Status, d.TotalActual, d.TotalPredicted, d.Count, d.OfficialCount])
        }};
        
        var layout = {{
            title: 'Index Potential Upside/Downside',
            yaxis: {{
                title: 'Mispricing (Predicted vs Actual)',
                tickformat: '.1%'
            }},
            xaxis: {{
                title: 'Index'
            }}
        }};
        
        Plotly.newPlot('plot', [trace], layout);
    </script>
</body>
</html>
"""
    output_file = "index_plot.html"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"✅ Index plot saved to {output_file}")
    try:
        os.startfile(output_file)
    except:
        pass

if __name__ == "__main__":
    create_index_plot()
