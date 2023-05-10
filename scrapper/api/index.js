const { google } = require("googleapis");
const path = require("path");

const spreadsheetId = "19UD2hi6sYo8E8U8Xmp5fqqbm1ZdYLxHKWkPV6eWQK0M";

const auth = new google.auth.GoogleAuth({
  keyFile: path.resolve("./api/credentials.json"),
  scopes: "https://www.googleapis.com/auth/spreadsheets"
});

const getGoogleSheets = async () => {
  const client = await auth.getClient();

  const googleSheets = google.sheets({ version: "v4", auth: client });
  return googleSheets;
};

const postHouses = async (newHouses, range) => {
  const googleSheets = await getGoogleSheets();

  await googleSheets.spreadsheets.values.append({
    auth,
    spreadsheetId,
    range,
    valueInputOption: "RAW",
    resource: {
      values: [...newHouses]
    }
  });
};

module.exports = {
  postHouses
};
