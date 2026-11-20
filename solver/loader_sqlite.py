"""POS sqlite export (tables don_hang, chi_tiet_don, thanh_toan, the_kho, ...)."""
import numpy as np
import pandas as pd

from loader import _num, status_of, unit_key
from tables import read_sqlite
from textnorm import norm
from values import clean_str, parse_bool, parse_datetime


def _df(tabs, name):
    if name not in tabs:
        return None
    cols, rows = tabs[name]
    return pd.DataFrame(rows, columns=cols)


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def _dts(s):
    return pd.to_datetime(s.map(lambda v: parse_datetime(v)), errors='coerce')


def load_sqlite(biz, path):
    T = read_sqlite(path)
    sp = _df(T, 'san_pham')
    prod = pd.DataFrame({
        'prod_id': _col(sp, 'MaHangHoa').map(clean_str),
        'name': _col(sp, 'TenGoi').map(clean_str),
        'category': _col(sp, 'DanhMuc').map(clean_str),
        'base_unit': _col(sp, 'DonViCoSo').map(unit_key),
        'cur_price': _col(sp, 'GiaBanLe').map(_num),
        'cost': _col(sp, 'GiaVonHang').map(_num),
        'active': _col(sp, 'ConKinhDoanh').map(parse_bool),
        'created': _dts(_col(sp, 'TaoLuc')),
    })
    prod['prod_key'] = prod['prod_id']
    tt = _df(T, 'thong_tin_thuoc')
    if tt is not None:
        m = tt.set_index('MaHangHoa')
        prod['ingredient'] = prod['prod_id'].map(m['ThanhPhanChinh'].map(clean_str))
        prod['rx'] = prod['prod_id'].map(m['BanTheoDon'].map(parse_bool))
    biz.products = prod
    biz.prod_names = dict(zip(prod['prod_key'], prod['name']))
    biz.prod_spellings = {k: [v] for k, v in biz.prod_names.items()}
    biz.extra['base_unit'] = dict(zip(prod['prod_key'], prod['base_unit']))

    units = {}
    dv = _df(T, 'don_vi_sp')
    if dv is not None:
        for pid, u, f in zip(dv['MaHangHoa'], dv['DonViTinh'], dv['HeSoQuyDoi']):
            units.setdefault(pid, {})[unit_key(u)] = _num(f)
    for pid, bu in zip(prod['prod_id'], prod['base_unit']):
        units.setdefault(pid, {})[bu] = 1.0
    biz.units = units

    dh = _df(T, 'don_hang')
    O = pd.DataFrame({
        'order_id': _col(dh, 'MaHoaDon').map(clean_str),
        'dt': _dts(_col(dh, 'ThoiGianLap')),
        'date_raw': _col(dh, 'NgayLap'),
        'status_raw': _col(dh, 'TinhTrang'),
        'cust_id': _col(dh, 'MaKhachHang').map(clean_str),
        'staff': _col(dh, 'MaNhanVien').map(clean_str),
        'shift_id': _col(dh, 'MaCaLam').map(clean_str),
        'channel': _col(dh, 'KenhBanHang').map(clean_str),
        'goods_total': _col(dh, 'TongTienHang').map(_num),
        'order_disc': _col(dh, 'ChietKhau').map(_num),
        'ship_fee': _col(dh, 'PhiVanChuyen').map(_num),
        'vat': _col(dh, 'ThueGTGT').map(_num),
        'payable': _col(dh, 'TongThanhToan').map(_num),
        'paid': _col(dh, 'KhachDaTra').map(_num),
        'debt': _col(dh, 'ConPhaiThu').map(_num),
        'pay_method': _col(dh, 'PhuongThucTT').map(clean_str),
        'promo': _col(dh, 'MaKhuyenMai').map(clean_str),
        'note': _col(dh, 'DienGiai').map(clean_str),
    })
    d1 = O['date_raw'].map(lambda v: parse_datetime(v))
    O['date'] = pd.to_datetime(d1, errors='coerce')
    O['date'] = O['date'].fillna(O['dt']).dt.normalize()
    O['status'] = O['status_raw'].map(lambda s: status_of(s) or 'completed')
    O = O.drop(columns=['date_raw'])

    ct = _df(T, 'chi_tiet_don')
    L = pd.DataFrame({
        'order_id': _col(ct, 'MaHoaDon').map(clean_str),
        'line_no': _col(ct, 'SoThuTu'),
        'prod_key': _col(ct, 'MaHangHoa').map(clean_str),
        'unit': _col(ct, 'DonViTinh').map(unit_key),
        'qty': _col(ct, 'SL').map(_num),
        'qty_base': _col(ct, 'SLQuyDoi').map(_num),
        'price': _col(ct, 'GiaDonVi').map(_num),
        'line_disc': _col(ct, 'ChietKhau').map(_num),
        'amount': _col(ct, 'TriGia').map(_num),
        'cost': _col(ct, 'GiaVonLucBan').map(_num),
    })
    L = L.drop_duplicates()
    L['prod_name'] = L['prod_key'].map(biz.prod_names)
    L['is_fee'] = False
    rev = L.groupby('order_id')['amount'].sum()
    O['rev'] = O['order_id'].map(rev).fillna(0.0)
    # returns recorded in tra_hang
    th = _df(T, 'tra_hang')
    if th is not None and len(th):
        R = pd.DataFrame({'order_id': th['MaHoaDon'].map(clean_str), 'dt': _dts(th['NgayGioTra']),
                          'refund': th['TienHoanTra'].map(_num), 'reason': th['NguyenNhan'].map(clean_str)})
        biz.returns = R
        m = O['order_id'].isin(set(R['order_id'])) & (O['status'] != 'cancelled')
        O.loc[m, 'status'] = 'returned'
    biz.orders = O
    biz.lines = L

    kh = _df(T, 'khach_hang')
    if kh is not None:
        biz.customers = pd.DataFrame({
            'cust_id': _col(kh, 'MaKhachHang').map(clean_str), 'name': _col(kh, 'TenGoi').map(clean_str),
            'phone': _col(kh, 'SoDienThoai').map(clean_str), 'group': _col(kh, 'LoaiKhachHang').map(clean_str),
            'gender': _col(kh, 'NamNu').map(clean_str), 'address': _col(kh, 'DiaChiLienHe').map(clean_str),
            'created': _dts(_col(kh, 'TaoLuc'))})
        if biz.customers['gender'].isna().all():
            biz.empty_cols.add('gioi tinh')
    nv = _df(T, 'nhan_vien')
    if nv is not None:
        biz.extra['staff'] = dict(zip(nv['MaNhanVien'], nv['TenGoi']))
    ncc = _df(T, 'nha_cung_cap')
    suppliers = dict(zip(ncc['MaNhaCungCap'], ncc['TenGoi'])) if ncc is not None else {}
    biz.extra['suppliers'] = suppliers

    tp = _df(T, 'thanh_toan')
    if tp is not None:
        biz.payments = pd.DataFrame({'order_id': tp['MaHoaDon'].map(clean_str), 'dt': _dts(tp['NgayGioThu']),
                                     'amount': tp['SoTienThu'].map(_num), 'method': tp['PhuongThuc'].map(clean_str),
                                     'pay_id': tp['MaPhieuThu'].map(clean_str)})

    pn = _df(T, 'phieu_nhap')
    cn = _df(T, 'chi_tiet_nhap')
    if pn is not None and cn is not None:
        H = pd.DataFrame({'rc_id': pn['MaPhieuNhap'].map(clean_str), 'supplier_id': pn['MaNhaCungCap'].map(clean_str),
                          'date': _dts(pn['NgayNhapKho']).dt.normalize(), 'status': pn['TinhTrang'].map(clean_str),
                          'rc_total': pn['TongGiaTriNhap'].map(_num), 'note': _col(pn, 'DienGiai').map(clean_str)})
        H['supplier'] = H['supplier_id'].map(suppliers)
        H['cancelled'] = H['status'].map(lambda s: bool(s) and ('huy' in norm(s) or 'cancel' in norm(s)))
        PL = pd.DataFrame({'rc_id': cn['MaPhieuNhap'].map(clean_str), 'line_no': cn['SoThuTu'],
                           'prod_key': cn['MaHangHoa'].map(clean_str), 'unit': cn['DonViTinh'].map(unit_key),
                           'qty': cn['SL'].map(_num), 'price': cn['DonGiaNhap'].map(_num),
                           'lot_id': _col(cn, 'MaLo').map(clean_str), 'expiry': _dts(_col(cn, 'HanSuDung'))})
        PL['amount'] = PL['qty'] * PL['price']
        PL['qty_base'] = [q * units.get(k, {}).get(u, 1.0) for q, k, u in zip(PL['qty'], PL['prod_key'], PL['unit'])]
        PL = PL.merge(H, on='rc_id', how='left')
        biz.purchases = PL
        biz.extra['rc_heads'] = H

    tk = _df(T, 'the_kho')
    if tk is not None:
        biz.stock_moves = pd.DataFrame({'prod_key': tk['MaHangHoa'].map(clean_str), 'dt': _dts(tk['NgayGioPhatSinh']),
                                        'kind': tk['LoaiGiaoDich'].map(clean_str), 'qty': tk['SLQuyDoi'].map(_num),
                                        'doc_type': tk['KieuChungTu'].map(clean_str),
                                        'doc_id': tk['MaChungTu'].map(clean_str), 'after': tk['TonCuoi'].map(_num),
                                        'seq': tk['MaGiaoDichKho']})
    ls = _df(T, 'lich_su_gia')
    if ls is not None:
        PR = pd.DataFrame({'prod_key': ls['MaHangHoa'].map(clean_str), 'date': _dts(ls['NgayApDung']).dt.normalize(),
                           'price': ls['GiaBanLe'].map(_num)})
        biz.prices = PR.sort_values(['prod_key', 'date']).reset_index(drop=True)
    lo = _df(T, 'lo_hang')
    if lo is not None:
        biz.lots = pd.DataFrame({'prod_key': lo['MaHangHoa'].map(clean_str), 'lot_id': lo['MaLo'].map(clean_str),
                                 'rc_id': lo['MaPhieuNhap'].map(clean_str), 'expiry': _dts(lo['HanSuDung']).dt.normalize(),
                                 'qty_in': lo['SLNhapQuyDoi'].map(_num), 'cost': lo['DonGiaNhap'].map(_num)})
    cl = _df(T, 'chi_tiet_don_lo')
    if cl is not None:
        biz.line_lots = pd.DataFrame({'order_id': cl['MaHoaDon'].map(clean_str), 'line_no': cl['SoThuTu'],
                                      'lot_id': cl['MaLo'].map(clean_str), 'qty': cl['SLQuyDoi'].map(_num),
                                      'ret': cl['SLTraQuyDoi'].map(_num).fillna(0)})
    biz.extra['tables'] = [(k, None, ['sqlite'], v[0]) for k, v in T.items()]
