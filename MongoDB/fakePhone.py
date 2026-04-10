from faker import Faker
import phonenumbers, random

fake = Faker()

def generate_valid_phone():
    while True:
        # Generate phone number with country code
        phone_number = fake.phone_number()
        #print(f"Generated phone number: {phone_number}")  # Debugging statement

        #PhoneNumberFormat.INTERNATIONAL
        #PhoneNumberFormat.NATIONAL
        #PhoneNumberFormat.E164
    

        try:
            parsed = phonenumbers.parse(phone_number, None)

            # Validate number
            if phonenumbers.is_valid_number(parsed) and phonenumbers.is_possible_number(parsed):
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )

        except phonenumbers.NumberParseException:
            continue


# Example usage
print('------------ :  ',generate_valid_phone())

# 🔹 Helper: Validate mobile number
def is_valid_mobile(parsed_number):
    return (
        phonenumbers.is_valid_number(parsed_number)
        and phonenumbers.number_type(parsed_number)
        == phonenumbers.PhoneNumberType.MOBILE
    )


# 🔹 Helper: Generate base valid mobile number
def generate_mobile_number(region="US"):
    while True:
        try:
            raw_number = fake.phone_number()
            parsed = phonenumbers.parse(raw_number, region)

            if is_valid_mobile(parsed):
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.E164
                )

        except:
            continue


# 🔹 PRIMARY NUMBER (with extension)
def generate_primary_number(region="US"):
    base_number = generate_mobile_number(region)

    # Generate random extension (3–5 digits)
    extension = str(random.randint(100, 99999))

    return {
        "type": "primary",
        "phone_number": base_number,
        "extension": extension,
        "formatted": f"{base_number} x{extension}"
    }


# 🔹 SECONDARY NUMBER (no extension)
def generate_secondary_number(region="US"):
    base_number = generate_mobile_number(region)

    return {
        "type": "secondary",
        "phone_number": base_number,
        "extension": None,
        "formatted": base_number
    }


# 🔹 Example Usage
if __name__ == "__main__":
    primary = generate_primary_number()
    secondary = generate_secondary_number()

    print("Primary:", primary)
    print("Secondary:", secondary)