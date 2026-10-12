"""Статическая визуализация и график из результатов эксперимента."""
import csv
import json
from pathlib import Path
import pypdfium2 as pdfium
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

def register_font():
    for path in (Path('C:/Windows/Fonts/arial.ttf'),
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.exists():
            pdfmetrics.registerFont(TTFont('PlotFont', str(path)))
            return
    raise FileNotFoundError('Нужен шрифт Arial или DejaVu Sans')

def raster(drawing, name):
    # Временная векторная версия служит только для качественного рендера.
    temporary = ROOT / 'tmp/pdf-review'
    temporary.mkdir(parents=True, exist_ok=True)
    path = temporary / (name + '.pdf')
    renderPDF.drawToFile(drawing, str(path))
    document = pdfium.PdfDocument(str(path))
    page = document[0]
    bitmap = page.render(scale=2)
    bitmap.to_pil().save(RESULTS / (name + '.png'))
    bitmap.close()
    page.close()
    document.close()

def label(drawing, x, y, text, size=16, color='#111111', anchor='start'):
    drawing.add(String(x, y, text, fontName='PlotFont', fontSize=size,
                       fillColor=colors.HexColor(color), textAnchor=anchor))

def chart(history):
    # Каждая точка - среднее отдельного последовательного блока 100 боёв.
    points = []
    for start in range(0, len(history), 100):
        block = history[start:start+100]
        points.append((int(block[-1]['episode']),
                       sum(float(row['reward']) for row in block)/len(block)))
    drawing = Drawing(720, 330)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 90, 65, 600, 225
    plot.data = [points]
    plot.xValueAxis.valueMin = 0
    plot.xValueAxis.valueMax = int(history[-1]['episode'])
    plot.xValueAxis.valueStep = max(1, int(history[-1]['episode'])//4)
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = -90, 60
    plot.yValueAxis.valueStep = 30
    for axis in (plot.xValueAxis, plot.yValueAxis):
        axis.labels.fontName = 'PlotFont'
        axis.labels.fontSize = 16
    plot.yValueAxis.visibleGrid = True
    plot.yValueAxis.gridStrokeColor = colors.HexColor('#CCCCCC')
    plot.lines[0].strokeColor = colors.HexColor('#222222')
    plot.lines[0].strokeWidth = 1.6
    plot.joinedLines = True
    drawing.add(plot)
    label(drawing, 90, 308, 'Средняя награда за 100 эпизодов', 18)
    label(drawing, 390, 18, 'Номер эпизода', 18, anchor='middle')
    raster(drawing, 'learning_curve')

def action_trace(demo):
    drawing = Drawing(720, 460)
    positions = [(15 + (index%2)*355, 347 - (index//2)*113)
                 for index in range(8)]
    for index, turn in enumerate(demo['turns']):
        x, y = positions[index]
        drawing.add(Rect(x, y, 335, 105, fillColor=colors.white,
                         strokeColor=colors.HexColor('#888888'), strokeWidth=0.7))
        label(drawing, x+12, y+80, f'Ход {turn["turn"]}. {turn["action_name"]}', 19)
        hp, enemy, energy, potion, _ = turn['next_state']
        label(drawing, x+12, y+56, f'Персонаж: {hp}/6', 16)
        label(drawing, x+180, y+56, f'Противник: {enemy}/8', 16)
        for bx, value, maximum, color in ((x+12, hp, 6, '#315F81'),
                                         (x+180, enemy, 8, '#9C4A43')):
            drawing.add(Rect(bx, y+36, 138, 12, fillColor=colors.HexColor('#EEEEEE'), strokeColor=None))
            if value:
                drawing.add(Rect(bx, y+36, 138*value/maximum, 12,
                                 fillColor=colors.HexColor(color), strokeColor=None))
        label(drawing, x+12, y+13, f'Энергия: {energy}/3; зелье: {potion}; R: {turn["reward"]:+d}', 15)
    x, y = positions[7]
    drawing.add(Rect(x, y, 335, 105, fillColor=colors.HexColor('#F3F3F3'),
                     strokeColor=colors.HexColor('#888888'), strokeWidth=0.7))
    label(drawing, x+167, y+70, 'ПОБЕДА', 22, anchor='middle')
    label(drawing, x+167, y+44, '7 ходов', 18, anchor='middle')
    label(drawing, x+167, y+18, f'Суммарная награда: {demo["total_reward"]}', 18, anchor='middle')
    raster(drawing, 'battle_trace')

def main():
    register_font()
    summary = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))
    with (RESULTS / 'training.csv').open(encoding='utf-8-sig', newline='') as stream:
        history = list(csv.DictReader(stream))
    chart(history)
    action_trace(summary['demo'])
    print('Static chart and action visualization generated')

if __name__ == '__main__':
    main()