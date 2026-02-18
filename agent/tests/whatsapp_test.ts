const url = 'https://graph.facebook.com/v17.0/106540352242922/messages';

const payload = {
  messaging_product: 'whatsapp',
  recipient_type: 'individual',
  to: '+919170439308',
  type: 'text',
  text: {
    preview_url: true,
    body: "Here's the info you requested! https://www.meta.com/quest/quest-3/",
  },
};

const res = await fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization:
      'Bearer EAARUJFKzIw0BQhEGkhnQo8n3u0QiZCvcmEVe3jnpkevuKxBoZCUSn8KH2jigUwCyVMwsbZAyZB0gdM6QpHfIhNObqG8POzF8qDvpwOfZBI1xLmUS5s3CyOVGzeyKLfyCDtsLjZC8goERAdsDRT9XwMd3JCposQA7PgU1kj568noaS9BEj2kt8ZAqTNTBvgFrXheDOy8z7ppCa96fJhdDjRte7IlfPMa6UQJXiBzX7uDqb5GZASO7sBr8ui6rUsmOiq66McwtLJld5llflaUTRZAUEuvzNXPBhdtQYP7o95s9b',
  },
  body: JSON.stringify(payload),
});

const text = await res.text();
console.log(res.status, res.statusText);
console.log(text);
