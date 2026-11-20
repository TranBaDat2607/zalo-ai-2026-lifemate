# Writing worked notes: instructions for the note writer

## What you are making

A small Vietnamese voice assistant (a 4B language model) answers the user's latest request by calling tools. Before
its first tool call it opens a thinking block and restates the request. We will train it to write one short **worked
note** right after that restatement. The note gives the reasoning that leads to the right calls. You write that note for
each training sample.

The note is written **at the first step of the final turn, before any tool runs in that turn**. It may use only what the
model can see at that moment:

- the current time and the list of coming days;
- the user profile;
- the earlier turns, including earlier tool calls and their results;
- the latest request.

It may **not** use anything that a tool call made in this final turn returns. In `gold_calls`, a value that only such a
result reveals is masked as `⟨từ kết quả gọi trước trong lượt này⟩`. Never guess it, and never write an ID, time or
amount that the model would only learn by reading. Instead, say which read gives it, for example
`contacts.search Lan để lấy contact_id`.

## Your input

Read only the input file named in your task, for example `notes/inputs/batch_000_p0.txt`. Do not open any other file
or folder: no data, no code, no other notes. Do not run scripts, and do not use the web or any other service. Each
sample block has these fields:

- `now` and `days`: the current date, time and weekday, and the coming days in D/M form.
- `profile` and `apps`: the user profile and the installed apps.
- `history`: earlier turns. Values that appear here are visible to the model.
- `request`: the latest request, which the model restates just before your note.
- `answer_type`:
  - `act`: some write is done;
  - `ask`: the answer ends by asking for a missing value;
  - `read_only`: only reads;
  - `nothing`: no tool is called.
- `gold_calls`: the correct calls of the final turn, in order.
- `tools`: the descriptions of the tools that gold uses.
- `must_include`: strings that **must appear in your note exactly as written**:
  - dates as D/M (8/11);
  - times as H:MM in 24-hour form (9:45, 16:00);
  - numbers as plain digits (200000);
  - IDs verbatim (item_646);
  - tool and parameter names verbatim.
- `may_depend_on_reads`: values that may depend on a read made in this turn.
  - Write such a value only if the request or the history gives it directly, for example a spoken absolute time.
  - If it is relative to something a read returns ("lùi 30 phút", "trước đó một tiếng"), describe the relation and
    leave the value out.
  - The same holds for a value computed from what a read returns: a count of items, a sum of amounts, a price times a
    quantity, or a time minus a route duration. Write the rule, not the result, for example
    `số phần = số thành viên nhóm trừ mình`, `tiền = 15 lít × giá đọc được` or `giờ đi = 16:00 trừ thời gian chạy`.

## What the note says

Keep only what helps the model get the calls right, in this order of importance:

1. **Dates and times resolved, with the arithmetic shown briefly.** For example:
   - `mười giờ kém mười lăm = 9:45`;
   - `T6 tuần sau = 13/11`;
   - `9:00 tới 9:45 = 45 phút`;
   - `trước 9:00 45 phút = 8:15`.
2. **Counts.** For example `2 lần`, `3 món`, `cả 2 đèn`.
3. **References to earlier turns.** What "đó", "ổng", "bả", "bữa đó", "cái kia" or "món vừa đặt" points to in the
   history, with its ID when the history shows it.
4. **The tool choice.** Always name the first gold tool exactly (for example `calendar.list_events`), and the others
   when useful.
   - Add a reason of 2 to 5 words only when the choice is not obvious, for example an unusual app, or two apps that
     could both fit.
   - Use the tool's description from `tools` for that reason.
5. **Asks.** Which required parameter is missing, then `sys.ask_missing_info` with the tool and the parameter names.
6. **Nothing.** For `answer_type: nothing`, say briefly why (no app can do it, small talk, the user takes it back, an
   invalid value, a condition that cannot be checked). The note must contain the exact words `không gọi tool`.
7. **Regional words.** Translate a Southern or Central word only when it matters for the tool or a value, for example
   `ni = nay`, `rứa = vậy`, `ổng = ông ấy`.

## Style and limits

- Write Vietnamese, terse and telegraphic. Arrows (→), `=`, `+`, `;` and parentheses are fine.
- Aim for **10 to 35 words**. The hard limits are 60 words and 100 tokens. Shorter is better: the model pays for every
  token at answer time, so do not pad.
- Use one line only. Do not use the characters `<` `>` `{` `}`, the word "think", or JSON.
- Do not restate the request, since the model already did that just before the note. Do not copy the call JSON. Do not
  mention default values.
- Do not write any date, time, number or ID that is not one of these:
  - a gold value;
  - visible to the model (time, profile, history, request);
  - a step of your own arithmetic.

  A checker rejects unexplained or contradicting values, and any value that only a result of this turn reveals.

## Examples (made up, not from the data)

| Situation | Note |
|---|---|
| History shows evt_12 "Họp chị Mai" on 8/11 at 9:00; request: "dời cuộc họp chị Mai sáng mai sang 3 giờ chiều"; gold `calendar.update_event` with event_id evt_12, start 8/11 15:00, end 8/11 16:00 | `Họp chị Mai mai 8/11 là evt_12 (lượt trước). 3 giờ chiều = 15:00, giữ 1 tiếng → 16:00. calendar.update_event.` |
| "đặt báo thức giúp anh" with no time; gold asks alarm.create [time] | `Chưa nói giờ báo thức → sys.ask_missing_info cho alarm.create, params time.` |
| "thôi khỏi, để anh tự làm" | `Người dùng rút lại yêu cầu nên không gọi tool.` |
| "nhắn Lan là anh tới trễ"; gold contacts.search then sms.send with a masked contact_id | `Cần contact_id của Lan: contacts.search trước, rồi sms.send.` |
| "lùi lịch nha sĩ chiều nay 30 phút"; start and end listed under may_depend_on_reads | `Chưa biết giờ lịch nha sĩ: calendar.list_events hôm nay 7/11, rồi lùi start và end 30 phút.` |

## Output

For each input part, write one JSONL file to the path named in your task, for example
`notes/parts/batch_000_p0.jsonl`. Put one line per sample, in input order, exactly `{"id": "train_XXXX", "note": "..."}`,
and use valid JSON. Write the whole file with the Write tool. When asked to fix notes, write only the listed ids to the
file path you are given.
