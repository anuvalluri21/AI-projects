Haymarket weather email agent

Requires Python 3.10 or newer. No extra packages are required.
Uses the National Weather Service API (weather.gov) for Haymarket, Virginia.
Each run retrieves the next four forecast periods, temperatures, wind,
precipitation chances, and a source link.

SETUP
1. Copy .env.example to .env in this folder.
2. Fill in the recipient, sender, and your email provider's SMTP settings.
   Use an SMTP credential or app password supported by your provider.
   Keep .env private. Existing environment variables override .env values.
3. Open a terminal in this folder.

PREVIEW (does not send)
python weather_agent.py

SEND ONCE
python weather_agent.py --send

SCHEDULING
No recurring schedule has been installed. Once you choose a sending time,
Windows Task Scheduler can run the script with --send. Use absolute paths
for the Python executable and script. The .env file is loaded relative to
the script, so the scheduler's working directory does not matter.
The computer must be running and have internet access at the scheduled time.

Failures return a nonzero exit code. Weather retrieval is retried up to three
times for transient failures. Email sends are not retried automatically,
to reduce duplicate delivery after an ambiguous SMTP failure.

NWS API documentation:
https://www.weather.gov/documentation/services-web-api
