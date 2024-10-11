import requests
import json
import prettytable


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

def get_data_from_bls():
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['CUUR0000SA0', 'SUUR0000SA0'], "startyear": "2020", "endyear": "2024"})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)
    for series in json_data['Results']['series']:
        x = prettytable.PrettyTable(["series id", "year", "period", "value", "footnotes"])
        seriesId = series['seriesID']
        for item in series['data']:
            year = item['year']
            period = item['period']
            value = item['value']
            footnotes = ""
            for footnote in item['footnotes']:
                if footnote:
                    footnotes = footnotes + footnote['text'] + ','
            if 'M01' <= period <= 'M12':
                x.add_row([seriesId, year, period, value, footnotes[0:-1]])
        output = open(seriesId + '.txt', 'w')
        output.write(x.get_string())
        output.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Team 90')
    get_data_from_bls()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
