"""Fetch the NWS forecast for Haymarket, VA and optionally email it."""
import argparse
import json
import os
from pathlib import Path
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def load_env():
    path = Path(__file__).with_name('.env')
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('\"').strip("'"))


def fetch(url):
    request = Request(url, headers={
        'User-Agent': os.getenv('NWS_USER_AGENT', 'HaymarketWeatherEmail/1.0'),
        'Accept': 'application/geo+json',
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def forecast_text():
    lat = float(os.getenv('LATITUDE', '38.8121'))
    lon = float(os.getenv('LONGITUDE', '-77.6364'))
    point = fetch(f'https://api.weather.gov/points/{lat:.4f},{lon:.4f}')
    url = point['properties']['forecast']
    if not url.startswith('https://api.weather.gov/'):
        raise ValueError('Unexpected forecast URL from NWS')
    props = fetch(url)['properties']
    periods = props.get('periods', [])
    if not periods:
        raise ValueError('NWS returned no forecast periods; email was not sent')
    lines = ['Weather forecast for Haymarket, Virginia',
             f"NWS updated: {props.get('updated', 'Unavailable')}", '']
    for period in periods[:4]:
        rain = (period.get('probabilityOfPrecipitation') or {}).get('value')
        lines.extend([
            f"{period['name']} ({period['startTime']}): {period['temperature']}°{period['temperatureUnit']}",
            f"Wind: {period['windDirection']} {period['windSpeed']}",
            f"Precipitation chance: {str(rain) + '%' if rain is not None else 'Unavailable'}",
            period['detailedForecast'], '',
        ])
    lines.append(f'Source: https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}')
    return '\n'.join(lines)


def send_email(text):
    required = ['SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'EMAIL_FROM', 'EMAIL_TO']
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError('Set these values in .env: ' + ', '.join(missing))
    message = EmailMessage()
    message['Subject'] = 'Weather forecast — Haymarket, VA'
    message['From'] = os.environ['EMAIL_FROM']
    message['To'] = os.environ['EMAIL_TO']
    message.set_content(text)
    mode = os.getenv('SMTP_SECURITY', 'starttls').lower()
    if mode not in ('ssl', 'starttls'):
        raise ValueError('SMTP_SECURITY must be ssl or starttls')
    port = int(os.getenv('SMTP_PORT') or ('465' if mode == 'ssl' else '587'))
    context = ssl.create_default_context()
    connection = (smtplib.SMTP_SSL(os.environ['SMTP_HOST'], port, timeout=30, context=context)
                  if mode == 'ssl' else smtplib.SMTP(os.environ['SMTP_HOST'], port, timeout=30))
    with connection as smtp:
        if mode == 'starttls':
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        smtp.login(os.environ['SMTP_USERNAME'], os.environ['SMTP_PASSWORD'])
        smtp.send_message(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--send', action='store_true', help='Send email; default is preview only')
    args = parser.parse_args()
    load_env()
    try:
        text = forecast_text()
        if args.send:
            send_email(text)
            print('Weather email sent.')
        else:
            print(text)
    except Exception as error:
        print(f'Weather agent failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
