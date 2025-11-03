# W&B Learning Rate Sweep Results

This notebook embeds and displays the W&B learning rate sweep experiment results directly from the W&B dashboard.

## 📊 W&B Project

**Project URL:** https://wandb.ai/tianweiyue-org/cs336-lr-sweep

## 🚀 Quick Start

### 1. Open the Notebook

```bash
jupyter notebook wandb_results.ipynb
```

Or if using VS Code, simply open the notebook and select your Python kernel.

### 2. Run All Cells

Execute all cells to:
- Embed the W&B dashboard directly in the notebook
- Display loss charts and run tables
- Access quick links to different views

## 📈 What's Included

The notebook embeds:

### 1. **Full W&B Workspace**
- Complete interactive dashboard
- All charts and visualizations
- Run comparisons

### 2. **Loss Visualization**
- Direct embed of the loss chart
- Training and validation loss curves
- All learning rates overlaid

### 3. **Runs Table**
- Sortable table of all experiments
- Filter and compare runs
- View metrics and hyperparameters

### 4. **Quick Links**
- Direct links to all W&B views
- Easy navigation to different sections
- Instructions for creating W&B Reports

## 🔧 Customization

### Change Project/Entity

Edit the URLs in the notebook cells:
```python
workspace_url = "https://wandb.ai/YOUR-ENTITY/YOUR-PROJECT/workspace"
```

### Add Screenshots

If iframes don't work, add screenshots:
```python
from IPython.display import Image
display(Image('wandb_screenshot.png', width=1000))
```

## 📊 Using W&B Dashboard

### Interactive Features
- **Zoom**: Click and drag on charts
- **Pan**: Hold shift and drag
- **Filter**: Click on run names to show/hide
- **Compare**: Select multiple runs to compare
- **Customize**: Add/remove metrics, change chart types

### Best Practices
- Use the **workspace** for interactive exploration
- Create **W&B Reports** for sharing and presentations
- Take **screenshots** for papers and static documents
- Use **table view** to sort and filter runs by metrics

## 🐛 Troubleshooting

### IFrames not loading
- **Browser security**: Some browsers block iframes. Click the links to open in new tabs instead.
- **Authentication**: Make sure you're logged into W&B in your browser.
- **Network**: Check your internet connection.

### Blank or error pages
- Verify the W&B project URL is correct
- Ensure you have access to the project
- Try opening the URL directly in your browser first

### Better alternatives
- **W&B Reports**: Create a report for better embedding and sharing
- **Screenshots**: Take screenshots from W&B and embed them as images
- **Direct links**: Use the quick links section to open views in new tabs

## 💡 Tips

1. **Open links in new tabs** for better viewing experience
2. **Use W&B Reports** for presentations and sharing
3. **Take screenshots** for static documents (papers, slides)
4. **Explore interactively** in the W&B dashboard
5. **Create custom views** by filtering and grouping runs

## 📚 Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Python API](https://docs.wandb.ai/ref/python)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## 🎯 Next Steps

After viewing the results:

1. **Identify the best learning rate** from the W&B dashboard
2. **Create a W&B Report** to document your findings
3. **Take screenshots** for your assignment or paper
4. **Analyze the loss curves** to understand training dynamics
5. **Compare runs** to see the effect of different learning rates

## 📝 Notes

- **Simple approach**: This notebook just embeds the W&B dashboard - no complex code needed!
- **Interactive**: All W&B features work in the embedded views
- **Always up-to-date**: Shows live data from your W&B project
- **Easy to customize**: Just change the URLs to point to your project

## 🎨 Creating W&B Reports (Recommended)

For the best presentation, create a W&B Report:

1. Go to https://wandb.ai/tianweiyue-org/cs336-lr-sweep
2. Click "Reports" → "Create Report"
3. Drag charts from your workspace into the report
4. Add markdown text to explain your findings
5. Share the report URL or export to PDF

**Benefits of W&B Reports:**
- ✅ Professional formatting
- ✅ Easy sharing and collaboration
- ✅ Version control
- ✅ Export to PDF
- ✅ Embed in other notebooks or websites

---

**Happy Analyzing! 📊✨**

