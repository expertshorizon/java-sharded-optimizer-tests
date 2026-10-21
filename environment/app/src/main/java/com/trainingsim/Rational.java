package com.trainingsim;

import java.math.BigInteger;

public final class Rational implements Comparable<Rational> {

    public static final Rational ZERO = new Rational(BigInteger.ZERO, BigInteger.ONE);
    public static final Rational ONE = new Rational(BigInteger.ONE, BigInteger.ONE);

    private final BigInteger numerator;
    private final BigInteger denominator;

    private Rational(BigInteger numerator, BigInteger denominator) {
        this.numerator = numerator;
        this.denominator = denominator;
    }

    public static Rational of(BigInteger numerator, BigInteger denominator) {
        if (denominator.signum() == 0) {
            throw new ArithmeticException("zero denominator");
        }
        BigInteger n = numerator;
        BigInteger d = denominator;
        if (n.signum() == 0) {
            return new Rational(BigInteger.ZERO, BigInteger.ONE);
        }
        BigInteger gcd = n.gcd(d);
        if (!gcd.equals(BigInteger.ONE)) {
            n = n.divide(gcd);
            d = d.divide(gcd);
        }
        return new Rational(n, d);
    }

    public static Rational ofLong(long value) {
        return of(BigInteger.valueOf(value), BigInteger.ONE);
    }

    public static Rational parseDecimal(String raw) {
        String text = raw.trim();
        boolean negative = false;
        if (text.startsWith("+")) {
            text = text.substring(1);
        } else if (text.startsWith("-")) {
            negative = true;
            text = text.substring(1);
        }
        int dot = text.indexOf('.');
        String digits;
        BigInteger scale;
        if (dot < 0) {
            digits = text;
            scale = BigInteger.ONE;
        } else {
            String intPart = text.substring(0, dot);
            String fracPart = text.substring(dot + 1);
            digits = intPart + fracPart;
            scale = BigInteger.TEN.pow(fracPart.length());
        }
        if (digits.isEmpty()) {
            digits = "0";
        }
        BigInteger n = new BigInteger(digits);
        if (negative) {
            n = n.negate();
        }
        return of(n, scale);
    }

    public static Rational parseFraction(String text) {
        int slash = text.indexOf('/');
        BigInteger n = new BigInteger(text.substring(0, slash));
        BigInteger d = new BigInteger(text.substring(slash + 1));
        return of(n, d);
    }

    public Rational add(Rational other) {
        BigInteger n = numerator.multiply(other.denominator).add(other.numerator.multiply(denominator));
        BigInteger d = denominator.multiply(other.denominator);
        return of(n, d);
    }

    public Rational sub(Rational other) {
        return add(other.negate());
    }

    public Rational mul(Rational other) {
        return of(numerator.multiply(other.numerator), denominator.multiply(other.denominator));
    }

    public Rational div(Rational other) {
        return of(numerator.multiply(other.denominator), denominator.multiply(other.numerator));
    }

    public Rational negate() {
        return of(numerator, denominator.negate());
    }

    public Rational abs() {
        return numerator.signum() < 0 ? negate() : this;
    }

    public int signum() {
        return numerator.signum();
    }

    @Override
    public int compareTo(Rational other) {
        BigInteger left = numerator.multiply(other.denominator);
        BigInteger right = other.numerator.multiply(denominator);
        return left.compareTo(right);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Rational)) {
            return false;
        }
        Rational other = (Rational) obj;
        return numerator.equals(other.numerator) && denominator.equals(other.denominator);
    }

    @Override
    public int hashCode() {
        return numerator.hashCode() * 31 + denominator.hashCode();
    }

    @Override
    public String toString() {
        return numerator.toString() + "/" + denominator.toString();
    }
}
