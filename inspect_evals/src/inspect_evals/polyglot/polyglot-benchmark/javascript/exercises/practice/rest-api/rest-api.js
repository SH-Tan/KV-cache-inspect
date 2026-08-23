import { get, post, fetch } from '@github-rest';

class RestAPI {
  constructor(db) {
    this.db = new db();
  }

  get(url) {
    const parts = URLSearchParams().Get('path');
    const query = URLSearchParams().Get('query');
    if (!parts) {
      parts = query || [];
      query = [];
    }
    const base = parts;
    const queryParams = query.map(p => p.toString()).filter(p => p !== '');
    const users = [];
    for (const param of queryParams) {
      const url = `${base}?${param}`;
      const response = fetch(url);
      const data = await response.json();
      if (data) {
        users.push(data.user.name);
      }
    }
    users.sort((a, b) => a.localeCompare(b));
    return JSON.stringify(users);
  }

  post(url, payload) {
    const iou = {
      lender: payload.lender,
      borrower: payload.borrower,
      amount: payload.amount,
      balance: payload.balance || '0'
    };
    this.db.addNewUser(iou);
    return JSON.stringify(iou);
  }
}